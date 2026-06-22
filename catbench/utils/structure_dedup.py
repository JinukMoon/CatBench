"""Frame-invariant structure deduplication primitives (CatBench 1.1.1).

Clean slabs and gas references are massively duplicated across reactions within a
dataset, but each copy is stored in its own frame (origin/orientation/atom order),
so a byte comparison treats them as distinct. These helpers identify *physically
identical* structures regardless of frame, so that:

  * storage can keep one copy and reference the rest (smaller files), and
  * a benchmark run can relax one representative and reuse the result (less GPU).

Two layers, by design:

  ``structure_fingerprint``  -- a fast, MLIP-agnostic, frame/order-invariant hash
      used only as a *candidate detector* (cheap to compute, may rarely collide).

  ``structures_equivalent``  -- an exact verifier: returns True only when one
      structure maps onto the other by a lattice translation + atom permutation
      within a tight tolerance (so their relaxed energies are identical). This is
      what actually authorizes a merge; the fingerprint only narrows candidates.

The reuse identity is geometry AND constraints: two slabs may share a geometry but
carry different FixAtoms, which would relax to different energies. ``dedup_key``
therefore fingerprints both the whole structure and its fixed-atom sub-structure.
"""

import hashlib

import numpy as np
from ase.constraints import FixAtoms
from ase.geometry import get_distances


def _fixed_indices(atoms):
    idx = []
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            idx.extend(int(i) for i in c.get_indices())
    return sorted(set(idx))


def structure_fingerprint(atoms, tol=0.01):
    """Frame- and order-invariant geometry hash (candidate detector).

    Composition + per-element-pair sorted pairwise distances (minimum-image when
    the cell is periodic), each rounded to ``tol`` Angstrom. Invariant to rigid
    translation/rotation and atom reordering; uses NO energy/MLIP/GPU.
    """
    sym = np.asarray(atoms.get_chemical_symbols())
    pos = atoms.get_positions()
    cell = atoms.cell
    pbc = bool(np.any(atoms.pbc))
    els = sorted(set(sym.tolist()))
    parts = []
    for i, a in enumerate(els):
        ia = np.where(sym == a)[0]
        for b in els[i:]:
            ib = np.where(sym == b)[0]
            if pbc:
                _, dmat = get_distances(pos[ia], pos[ib], cell=cell, pbc=True)
            else:
                dmat = np.linalg.norm(pos[ia][:, None, :] - pos[ib][None, :, :], axis=2)
            if a == b:
                vals = dmat[np.triu_indices(len(ia), 1)]
            else:
                vals = dmat.ravel()
            vals = np.sort(np.round(np.asarray(vals) / tol) * tol)
            parts.append("%s%s:" % (a, b) + ",".join("%.3f" % v for v in vals))
    comp = "".join("%s%d" % (e, int((sym == e).sum())) for e in els)
    digest = hashlib.md5((comp + "|" + "|".join(parts)).encode()).hexdigest()
    return digest


def structures_equivalent(a, b, tol=1e-3):
    """True iff ``b`` equals ``a`` up to a lattice translation + atom permutation,
    within ``tol`` Angstrom (an exact frame match => identical relaxed energy).

    Conservative: only returns True for genuinely identical-up-to-frame structures.
    False-negatives (e.g. a pure rotation, or atoms wrapped differently than the
    anchor handles) are harmless -- they merely forgo a dedup, never corrupt one.
    """
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True  # two empty structures are trivially equivalent (no sa[0] anchor)
    sa = np.asarray(a.get_chemical_symbols())
    sb = np.asarray(b.get_chemical_symbols())
    if sorted(sa.tolist()) != sorted(sb.tolist()):
        return False
    if np.abs(np.asarray(a.cell) - np.asarray(b.cell)).max() > 0.01:
        return False
    pa = a.get_positions()
    pb = b.get_positions()
    cell = a.cell
    pbc = bool(np.any(a.pbc))
    # Anchor atom 0 of `a` onto each same-element atom of `b`; the implied
    # translation must yield a within-tol bijection between the two atom sets.
    for j in np.where(sb == sa[0])[0]:
        shifted = pa + (pb[j] - pa[0])
        if pbc:
            _, dmat = get_distances(shifted, pb, cell=cell, pbc=True)
        else:
            dmat = np.linalg.norm(shifted[:, None, :] - pb[None, :, :], axis=2)
        used = set()
        ok = True
        for ai in range(len(a)):
            matched = False
            for bi in np.argsort(dmat[ai]):
                if dmat[ai, bi] > tol:
                    break
                if bi in used or sb[bi] != sa[ai]:
                    continue
                used.add(int(bi))
                matched = True
                break
            if not matched:
                ok = False
                break
        if ok and len(used) == len(a):
            return True
    return False


def reuse_key(atoms, energy_ref, tol=0.01):
    """Composite slab/structure identity for cache & dedup: the stored DFT
    energy (primary, exact, frame-invariant) AND the geometry+fix fingerprint
    must BOTH match to count as the same structure.

    The two signals are independent cross-checks: a coincidental fingerprint
    collision is caught by a differing DFT energy, and a coincidental energy
    match is caught by a differing fingerprint -- so a wrong reuse is essentially
    impossible. On real data (Mamun, 45130 slabs) the two agree perfectly (1916
    unique by either => identical grouping), so requiring both costs no dedup.

    ``energy_ref`` is the per-structure DFT energy stored in the dataset
    (``raw[system]["energy_ref"]``). If it is missing, falls back to geometry+fix
    only.
    """
    geo = dedup_key(atoms, tol=tol)
    # A missing or non-finite (NaN/inf) energy can't serve as a cross-check, so
    # fall back to geometry+fix only and mark the key explicitly (never emit a
    # "nan|" prefix that silently disables the energy check).
    if energy_ref is None or not np.isfinite(float(energy_ref)):
        return "noE|" + geo
    return ("%.6f|" % round(float(energy_ref), 6)) + geo


def dedup_key(atoms, tol=0.01):
    """Geometry+constraint identity hash used to bucket reuse candidates.

    Combines the whole-structure fingerprint with the fingerprint of the
    fixed-atom sub-structure, so two slabs collide only when BOTH the geometry
    and the fixed-atom arrangement match. Frame/order invariant. Still only a
    *candidate* key -- callers verify with ``structures_equivalent`` (and a
    matching fixed set) before merging.
    """
    fp_all = structure_fingerprint(atoms, tol=tol)
    fidx = _fixed_indices(atoms)
    if fidx:
        sub = atoms[fidx]
        fp_fix = structure_fingerprint(sub, tol=tol)
    else:
        fp_fix = "nofix"
    return fp_all + "/" + fp_fix
