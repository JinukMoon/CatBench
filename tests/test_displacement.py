"""Tests pinning the find_mic displacement behavior (C1/C2) introduced in 1.1.0.

These guard against silent regressions in the minimum-image-convention (MIC)
correction used by ``calc_displacement`` and the substrate-displacement path.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.geometry import find_mic

from catbench.utils.calculation_utils import calc_displacement


def _orthogonal_atoms():
    return Atoms(
        "Cu3",
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )


def test_orthogonal_normal_displacement_matches_hand_norm():
    """Common case (orthogonal cell, no wrapping) must be unchanged: the
    mic-corrected displacement equals the plain hand-computed norm."""
    a1 = _orthogonal_atoms()
    a2 = a1.copy()
    pos = a2.get_positions()
    pos[1] = pos[1] + np.array([0.3, 0.4, 0.0])  # norm = 0.5
    pos[2] = pos[2] + np.array([0.0, 0.0, 1.2])  # norm = 1.2
    a2.set_positions(pos)

    result = calc_displacement(a1, a2)

    # max_disp must equal the largest hand-computed norm.
    assert result["max_disp"] == pytest.approx(1.2)
    # The full per-atom magnitudes match the trivial difference norms.
    expected = np.linalg.norm(a2.get_positions() - a1.get_positions(), axis=1)
    assert result["mae_mobile"] == pytest.approx(np.mean(expected))
    assert result["rmsd_mobile"] == pytest.approx(np.sqrt(np.mean(expected**2)))


def test_hexagonal_wrapped_atom_uses_mic_not_diagonal():
    """C1 bug fix: in a non-orthogonal (hexagonal, gamma=120) cell, an atom that
    wrapped across a periodic boundary must report the small TRUE displacement,
    not the inflated diagonal-only difference."""
    a = 3.0
    cell = [[a, 0.0, 0.0], [-a / 2.0, a * np.sqrt(3) / 2.0, 0.0], [0.0, 0.0, 20.0]]
    a1 = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 5.0]],
        cell=cell,
        pbc=True,
    )
    a2 = a1.copy()
    pos = a2.get_positions()
    # Move atom 0 by a full lattice vector (a wrap) plus a tiny real shift of 0.1.
    true_shift = np.array([0.1, 0.0, 0.0])
    pos[0] = pos[0] + np.array(cell[0]) + true_shift
    a2.set_positions(pos)

    result = calc_displacement(a1, a2)

    inflated = np.linalg.norm(np.array(cell[0]) + true_shift)
    assert inflated > 3.0  # sanity: the naive (no-mic) value would be large
    # MIC must recover the small true displacement.
    assert result["max_disp"] == pytest.approx(0.1, abs=1e-6)
    assert result["max_disp"] < 1.0


def test_calc_displacement_excludes_fixed_indices_for_mae():
    """Mobile-atom statistics exclude fixed indices (complement set)."""
    a1 = _orthogonal_atoms()
    a2 = a1.copy()
    pos = a2.get_positions()
    pos[0] = pos[0] + np.array([0.0, 0.0, 10.0])  # huge move on a "fixed" atom
    pos[1] = pos[1] + np.array([0.0, 0.0, 0.2])
    a2.set_positions(pos)

    result = calc_displacement(a1, a2, fixed_indices=[0])

    # Index 0 is fixed -> excluded from mae/rmsd; only atoms 1 (0.2) and 2 (0.0).
    assert result["mae_mobile"] == pytest.approx(np.mean([0.2, 0.0]))
    # But max_disp still considers every atom.
    assert result["max_disp"] == pytest.approx(10.0)


def test_substrate_displacement_excludes_nontrailing_adsorbate():
    """C2: substrate atoms are indexed in adslab space and the adsorbate is
    excluded correctly even when its index is NOT trailing.

    This reproduces the exact logic of
    AdsorptionCalculation._calculate_substrate_displacement so the contract is
    pinned without instantiating the heavy calculation object.
    """
    adslab_initial = Atoms(
        "Cu4",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    adslab_final = adslab_initial.copy()
    pos = adslab_final.get_positions()
    pos[1] = pos[1] + np.array([0.0, 0.0, 5.0])  # adsorbate at index 1 moves a lot
    adslab_final.set_positions(pos)

    adsorbate_indices = [1]  # NOT trailing
    substrate_indices = [
        i for i in range(len(adslab_initial)) if i not in adsorbate_indices
    ]
    assert substrate_indices == [0, 2, 3]

    diffs = (
        adslab_final.get_positions()[substrate_indices]
        - adslab_initial.get_positions()[substrate_indices]
    )
    mic_diffs, _ = find_mic(diffs, adslab_initial.cell, adslab_initial.pbc)
    substrate_max = float(np.linalg.norm(mic_diffs, axis=1).max())

    # Substrate atoms did not move; the large adsorbate move must be excluded.
    assert substrate_max == pytest.approx(0.0, abs=1e-9)


def test_substrate_displacement_method_on_instance():
    """Exercise the actual bound method on a lightweight AdsorptionCalculation
    instance (no calculation run, just the displacement helper)."""
    from ase.calculators.emt import EMT
    from catbench.adsorption import AdsorptionCalculation

    calc = AdsorptionCalculation([EMT()], mlip_name="x", benchmark="y")

    adslab_initial = Atoms(
        "Cu4",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    adslab_final = adslab_initial.copy()
    pos = adslab_final.get_positions()
    pos[0] = pos[0] + np.array([0.0, 0.0, 0.3])  # substrate atom 0 moves 0.3
    pos[1] = pos[1] + np.array([0.0, 0.0, 8.0])  # adsorbate (index 1) moves a lot
    adslab_final.set_positions(pos)

    out = calc._calculate_substrate_displacement(
        slab_initial=None,
        slab_final=None,
        adslab_initial=adslab_initial,
        adslab_final=adslab_final,
        adsorbate_indices=[1],
    )
    assert out == pytest.approx(0.3, abs=1e-6)
