"""Tests pinning constraint/FixAtoms semantics (C2/C4) and CatHub constraint
parsing introduced in 1.1.0. All offline (in-memory fixtures only)."""

import io
import json

import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write

from catbench.adsorption.data.cathub import aseify_reactions
from catbench.utils.calculation_utils import get_fixed_indices
from catbench.utils.data_utils import load_catbench_json, save_catbench_json
from catbench.utils.io_utils import get_calculation_settings


def _ase_json(atoms):
    buf = io.StringIO()
    write(buf, atoms, format="json")
    return buf.getvalue()


def _sample_atoms(n=4):
    positions = [[float(i), 0.0, float(i)] for i in range(n)]
    return Atoms(f"Cu{n}", positions=positions, cell=[20.0, 20.0, 20.0], pbc=True)


# --------------------------------------------------------------------------- #
# get_fixed_indices / rate semantics
# --------------------------------------------------------------------------- #
def test_get_fixed_indices_rate_none_uses_own_fixatoms_sorted():
    atoms = _sample_atoms(4)
    atoms.set_constraint(FixAtoms(indices=[2, 0]))  # deliberately unsorted
    assert get_fixed_indices(atoms, None) == [0, 2]


def test_get_fixed_indices_rate_float_uses_legacy_z():
    atoms = _sample_atoms(4)  # z positions 0,1,2,3
    atoms.set_constraint(FixAtoms(indices=[3]))  # must be ignored by legacy path
    # Fix atoms with z < z_target=1.5 -> atoms 0 (z=0) and 1 (z=1).
    assert get_fixed_indices(atoms, rate=0.5, z_target=1.5) == [0, 1]


def test_get_fixed_indices_no_constraints_rate_none_returns_empty():
    atoms = _sample_atoms(2)
    assert get_fixed_indices(atoms, None) == []


def test_get_fixed_indices_dedups_multiple_constraints():
    atoms = _sample_atoms(4)
    atoms.set_constraint([FixAtoms(indices=[0, 1]), FixAtoms(indices=[1, 3])])
    assert get_fixed_indices(atoms, None) == [0, 1, 3]


# --------------------------------------------------------------------------- #
# CatHub constraint parsing (offline aseify_reactions)
# --------------------------------------------------------------------------- #
def test_aseify_reactions_attaches_fixatoms_from_constraints_json():
    atoms = _sample_atoms(4)
    constraints_json = json.dumps(
        [{"name": "FixAtoms", "kwargs": {"indices": [0, 1, 2]}}]
    )
    reactions = [
        {
            "reactionSystems": [
                {
                    "name": "star",
                    "systems": {
                        "energy": -1.0,
                        "constraints": constraints_json,
                        "InputFile": _ase_json(atoms),
                    },
                }
            ]
        }
    ]

    aseify_reactions(reactions)

    star = reactions[0]["reactionSystems"]["star"]["atoms"]
    fixed = [c for c in star.constraints if isinstance(c, FixAtoms)]
    assert len(fixed) == 1
    assert sorted(int(i) for i in fixed[0].get_indices()) == [0, 1, 2]


def test_aseify_reactions_none_constraints_yields_no_fixatoms():
    atoms = _sample_atoms(3)
    reactions = [
        {
            "reactionSystems": [
                {
                    "name": "gas",
                    "systems": {
                        "energy": -0.5,
                        "constraints": None,
                        "InputFile": _ase_json(atoms),
                    },
                }
            ]
        }
    ]

    aseify_reactions(reactions)

    gas = reactions[0]["reactionSystems"]["gas"]["atoms"]
    assert [c for c in gas.constraints if isinstance(c, FixAtoms)] == []


# --------------------------------------------------------------------------- #
# FixAtoms round-trip through save_catbench_json -> load_catbench_json
# --------------------------------------------------------------------------- #
def test_fixatoms_survives_catbench_json_roundtrip(tmp_path):
    atoms = _sample_atoms(4)
    atoms.set_constraint(FixAtoms(indices=[0, 1, 2]))

    data = {
        "rxn_0": {
            "raw": {
                "star": {"atoms": atoms, "energy_ref": -1.0, "stoi": 1},
            }
        }
    }

    path = tmp_path / "data.json"
    save_catbench_json(data, str(path))
    loaded = load_catbench_json(str(path))

    restored = loaded["rxn_0"]["raw"]["star"]["atoms"]
    fixed = [c for c in restored.constraints if isinstance(c, FixAtoms)]
    assert len(fixed) == 1
    assert sorted(int(i) for i in fixed[0].get_indices()) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# constraint_mode + version stamp (C4)
# --------------------------------------------------------------------------- #
def test_get_calculation_settings_rate_none_is_fixatoms():
    settings = get_calculation_settings({"rate": None})
    assert settings["constraint_mode"] == "fixatoms"


def test_get_calculation_settings_float_rate_is_legacy_z():
    settings = get_calculation_settings({"rate": 0.5})
    assert settings["constraint_mode"] == "legacy_z"


def test_get_calculation_settings_includes_version_stamp():
    settings = get_calculation_settings({"rate": None})
    assert "catbench_version" in settings
    assert isinstance(settings["catbench_version"], str)
    assert settings["catbench_version"]
