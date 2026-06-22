"""C1 — frame-invariant fingerprint + equivalence (CatBench 1.1.1 dedup)."""
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from catbench.utils.structure_dedup import (
    structure_fingerprint,
    structures_equivalent,
    dedup_key,
    reuse_key,
)


def _slab():
    # small 2-element periodic "slab"
    rng = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
           [0, 0, 1.2], [1, 0, 1.2], [0, 1, 1.2], [1, 1, 1.2]]
    return Atoms("Cu4Pt4", positions=rng, cell=[2, 2, 12], pbc=True)


def test_identity():
    a = _slab()
    assert structure_fingerprint(a) == structure_fingerprint(a.copy())


def test_translation_invariant():
    a = _slab()
    b = a.copy()
    b.translate([0.37, -1.1, 0.0])
    b.wrap()
    assert structure_fingerprint(a) == structure_fingerprint(b)


def test_rotation_invariant():
    a = Atoms("Cu2Pt2", positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 2]],
              cell=[10, 10, 10], pbc=False)
    b = a.copy()
    b.rotate(40, "z")
    b.rotate(15, "x")
    assert structure_fingerprint(a) == structure_fingerprint(b)


def test_permutation_invariant():
    a = _slab()
    b = a.copy()
    order = [5, 0, 7, 2, 1, 4, 3, 6]
    b = b[order]
    assert structure_fingerprint(a) == structure_fingerprint(b)


def test_near_duplicate_does_not_collide():
    a = _slab()
    b = a.copy()
    p = b.get_positions()
    p[3] += [0.2, 0.0, 0.0]  # nudge one atom well beyond tol
    b.set_positions(p)
    assert structure_fingerprint(a) != structure_fingerprint(b)


def test_composition_discriminates():
    a = _slab()
    b = a.copy()
    b.numbers[0] = 47  # Cu -> Ag
    assert structure_fingerprint(a) != structure_fingerprint(b)


def test_deterministic_across_calls():
    a = _slab()
    fps = {structure_fingerprint(a) for _ in range(5)}
    assert len(fps) == 1


def test_structures_equivalent_translation_and_permutation():
    a = _slab()
    b = a.copy()
    b.translate([0.5, 0.5, 0.0])
    b.wrap()
    b = b[[7, 1, 3, 5, 0, 2, 4, 6]]
    assert structures_equivalent(a, b)


def test_structures_equivalent_rejects_different():
    a = _slab()
    b = a.copy()
    p = b.get_positions()
    p[2] += [0.3, 0.1, 0.0]
    b.set_positions(p)
    assert not structures_equivalent(a, b)


def test_dedup_key_same_geometry_same_fix_collide():
    a = _slab()
    a.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    b = a.copy()
    b.translate([0.5, 0.0, 0.0])
    b.wrap()
    assert dedup_key(a) == dedup_key(b)


def test_dedup_key_same_geometry_different_fix_split():
    a = _slab()
    a.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    b = _slab()
    b.set_constraint(FixAtoms(indices=[0, 1]))  # different fix set
    assert dedup_key(a) != dedup_key(b)


# --- reuse_key: DFT energy AND fingerprint must both match ------------------- #
def test_reuse_key_same_energy_and_geometry_collide():
    a = _slab()
    a.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    b = a.copy()
    b.translate([0.5, 0.0, 0.0])
    b.wrap()
    assert reuse_key(a, -123.456789) == reuse_key(b, -123.456789)


def test_reuse_key_same_geometry_different_energy_split():
    # fingerprint coincides but DFT energy differs -> must NOT merge
    a = _slab()
    a.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    assert reuse_key(a, -123.4567) != reuse_key(a, -123.9999)


def test_reuse_key_same_energy_different_geometry_split():
    # energy coincides but geometry differs -> must NOT merge
    a = _slab()
    b = a.copy()
    p = b.get_positions()
    p[3] += [0.25, 0.0, 0.0]
    b.set_positions(p)
    assert reuse_key(a, -50.0) != reuse_key(b, -50.0)


def test_reuse_key_missing_energy_falls_back_to_geometry():
    a = _slab()
    assert reuse_key(a, None) == "noE|" + dedup_key(a)
