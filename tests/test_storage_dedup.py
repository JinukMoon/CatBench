"""C2 — storage dedup (ref markers + backward-compatible resolve), 1.1.1."""
import io
import json
import os

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from catbench.utils.data_utils import save_catbench_json, load_catbench_json, _write_catbench_json
from catbench.utils.structure_dedup import structures_equivalent
from catbench.utils.calculation_utils import get_fixed_indices


def _slab(shift=(0.0, 0.0, 0.0)):
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                    [0, 0, 1.2], [1, 0, 1.2], [0, 1, 1.2], [1, 1, 1.2]], float)
    a = Atoms("Cu4Pt4", positions=pos, cell=[2, 2, 12], pbc=True)
    a.translate(shift)
    a.wrap()
    a.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    return a


def _gas():
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=True)


def _data_dict():
    # Two reactions sharing a frame-equivalent slab + the same H2 gas reference.
    d = {}
    for i, sh in enumerate([(0, 0, 0), (0.5, 0.5, 0.0)]):
        ads = _slab(sh)  # adslab variant (distinct per reaction so it won't dedup)
        ads += Atoms("O", positions=[[0.5 + 0.01 * i, 0.5, 2.0]])
        d["rxn%d" % i] = {
            "raw": {
                "star": {"atoms": _slab(sh), "energy_ref": -1.0, "stoi": -1},
                "Ostar": {"atoms": ads, "energy_ref": 0.0, "stoi": 1},
                "H2gas": {"atoms": _gas(), "energy_ref": -7.0, "stoi": -0.5},
            },
            "ref_ads_eng": 0.3,
            "adsorbate_indices": [8],
            "constraint_source": "deposited",
        }
    return d


def _atoms_equal(a, b):
    return (np.allclose(a.get_positions(), b.get_positions(), atol=1e-8)
            and list(a.numbers) == list(b.numbers)
            and np.allclose(a.cell, b.cell, atol=1e-8))


def test_dedup_roundtrip_preserves_every_structure(tmp_path):
    orig = _data_dict()
    # keep a pristine copy of original atoms for comparison
    import copy
    ref_atoms = {rk: {sk: sv["atoms"].copy() for sk, sv in rv["raw"].items()}
                 for rk, rv in orig.items()}

    p = str(tmp_path / "ded.json")
    save_catbench_json(copy.deepcopy(orig), p)

    raw = json.load(open(p))
    assert "_structures" in raw                      # dedup happened
    # every system now references the structure store, none stores atoms_json
    for rk, rv in raw.items():
        if rk == "_structures":
            continue
        for sv in rv["raw"].values():
            assert "ref" in sv and "atoms_json" not in sv

    loaded = load_catbench_json(p)
    assert set(loaded) == set(orig)                  # _structures not seen as a reaction
    for rk in orig:
        for sk in orig[rk]["raw"]:
            la = loaded[rk]["raw"][sk]["atoms"]
            oa = ref_atoms[rk][sk]
            # Dedup replaces a structure with a frame-equivalent canonical copy:
            # the guarantee is physical identity (=> identical relaxed energy),
            # not byte-identical coordinates.
            assert structures_equivalent(la, oa)
            # the fixed-atom set (which determines the relaxed energy) is preserved
            assert get_fixed_indices(la) == get_fixed_indices(oa)
            # non-geometry fields preserved
            assert loaded[rk]["raw"][sk]["energy_ref"] == orig[rk]["raw"][sk]["energy_ref"]
        assert loaded[rk]["adsorbate_indices"] == orig[rk]["adsorbate_indices"]
        assert loaded[rk]["constraint_source"] == "deposited"


def test_dedup_actually_shares_slab_and_gas(tmp_path):
    p = str(tmp_path / "ded.json")
    save_catbench_json(_data_dict(), p)
    raw = json.load(open(p))
    # 2 frame-equivalent slabs + 2 identical gas -> far fewer stored structures
    # than the 6 raw systems (2 star + 2 adslab + 2 gas).
    n_stored = len(raw["_structures"])
    assert n_stored <= 4  # slab(1) + gas(1) + 2 distinct adslabs
    # both reactions' star + gas point at the same stored hash
    refs = lambda k: {rk: raw[rk]["raw"][k]["ref"] for rk in raw if rk != "_structures"}
    assert len(set(refs("star").values())) == 1
    assert len(set(refs("H2gas").values())) == 1


def test_legacy_file_without_structures_still_loads(tmp_path):
    p = str(tmp_path / "legacy.json")
    _write_catbench_json(_data_dict(), p, dedup=False)   # legacy 1.1.0-style
    raw = json.load(open(p))
    assert "_structures" not in raw
    for rk, rv in raw.items():
        for sv in rv["raw"].values():
            assert "atoms_json" in sv and "ref" not in sv
    loaded = load_catbench_json(p)                     # must still load
    assert set(loaded) == {"rxn0", "rxn1"}
    assert loaded["rxn0"]["raw"]["star"]["atoms"] is not None


def test_dedup_file_is_smaller(tmp_path):
    a = str(tmp_path / "ded.json")
    b = str(tmp_path / "leg.json")
    save_catbench_json(_data_dict(), a)
    _write_catbench_json(_data_dict(), b, dedup=False)
    assert os.path.getsize(a) < os.path.getsize(b)
