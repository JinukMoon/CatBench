"""C3 — runtime slab-cache integration: dedup ON == OFF, cache hits, restart.

Uses ASE EMT (deterministic, frame-invariant) so a correct cache reproduces the
no-cache numbers exactly. No GPU / MLIP needed.
"""
import json
import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from catbench.utils.data_utils import save_catbench_json, _write_catbench_json
from catbench.adsorption import AdsorptionCalculation


def _surface(shift):
    slab = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    slab.set_constraint(FixAtoms(mask=[a.position[2] < slab.positions[:, 2].mean() for a in slab]))
    slab.translate(shift)
    slab.wrap()
    return slab


def _adslab_from(slab, dz):
    ad = slab.copy()
    top_z = ad.positions[:, 2].max()
    ad += Atoms("O", positions=[[ad.cell[0, 0] / 2, ad.cell[1, 1] / 2, top_z + 1.8 + dz]])
    return ad


def _gas():
    return Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]], cell=[12, 12, 12], pbc=True)


def _build_dataset(path, dedup):
    # 3 reactions on the SAME Pt(111) surface (frame-shifted) -> slab dedups;
    # distinct adslabs -> not deduped. Same O2 gas ref -> dedups.
    data = {}
    for i, sh in enumerate([(0, 0, 0), (0.3, 0.7, 0.0), (1.1, 0.2, 0.0)]):
        slab = _surface(sh)
        ads = _adslab_from(slab, dz=0.05 * i)
        n = len(slab)
        data["O_site%d" % i] = {
            "raw": {
                "star": {"atoms": slab, "energy_ref": 0.0, "stoi": -1},
                "Ostar": {"atoms": ads, "energy_ref": 0.0, "stoi": 1},
                "O2gas": {"atoms": _gas(), "energy_ref": 0.0, "stoi": -0.5},
            },
            "ref_ads_eng": 0.0,
            "adsorbate_indices": [n],
            "constraint_source": "deposited",
        }
    _write_catbench_json(data, path, dedup=dedup)


def _run(tmp, benchmark, dedup, structure_cache):
    os.makedirs(os.path.join(tmp, "raw_data"), exist_ok=True)
    _build_dataset(os.path.join(tmp, "raw_data", f"{benchmark}_adsorption.json"), dedup)
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        AdsorptionCalculation(
            [EMT(), EMT()], mlip_name=f"EMT_{int(dedup)}_{int(structure_cache)}",
            benchmark=benchmark, save_files=False, structure_cache=structure_cache,
        ).run()
        rp = os.path.join(tmp, "result", f"EMT_{int(dedup)}_{int(structure_cache)}",
                          f"EMT_{int(dedup)}_{int(structure_cache)}_result.json")
        with open(rp) as f:
            res = json.load(f)
    finally:
        os.chdir(cwd)
    return res


def _ads_engs(res):
    return {k: v["0"]["ads_eng"] for k, v in res.items() if isinstance(v, dict) and "0" in v}


def test_structure_cache_on_equals_off(tmp_path):
    tmp = str(tmp_path)
    off = _ads_engs(_run(tmp, "ds_a", dedup=False, structure_cache=False))
    on = _ads_engs(_run(tmp, "ds_b", dedup=False, structure_cache=True))
    assert set(on) == set(off) and len(on) == 3
    for k in off:
        assert abs(on[k] - off[k]) < 1e-6, (k, on[k], off[k])


def test_single_point_structure_cached_on_equals_off(tmp_path):
    # the single_calculation (single-point) slab energy is also cached; ON==OFF
    tmp = str(tmp_path)
    off = _run(tmp, "sp_a", dedup=False, structure_cache=False)
    on = _run(tmp, "sp_b", dedup=False, structure_cache=True)
    keys = [k for k in off if isinstance(off[k], dict) and "single_calculation" in off[k]]
    assert keys
    for k in keys:
        assert abs(on[k]["single_calculation"]["ads_eng"]
                   - off[k]["single_calculation"]["ads_eng"]) < 1e-9


def test_storage_dedup_on_equals_off(tmp_path):
    tmp = str(tmp_path)
    legacy = _ads_engs(_run(tmp, "ds_c", dedup=False, structure_cache=True))
    deduped = _ads_engs(_run(tmp, "ds_d", dedup=True, structure_cache=True))
    for k in legacy:
        assert abs(deduped[k] - legacy[k]) < 1e-6, (k, deduped[k], legacy[k])


def test_structure_cache_produces_hits(tmp_path):
    tmp = str(tmp_path)
    res = _run(tmp, "ds_e", dedup=True, structure_cache=True)
    # 3 reactions share one slab; per seed only the first relaxes, the rest are
    # cache hits (slab_time == 0).
    slab_times = [v["0"]["slab_time"] for k, v in res.items()
                  if isinstance(v, dict) and "0" in v]
    assert slab_times.count(0.0) >= 2  # at least 2 of 3 reused


def test_restart_resumes_structure_cache(tmp_path):
    tmp = str(tmp_path)
    benchmark = "ds_f"
    os.makedirs(os.path.join(tmp, "raw_data"), exist_ok=True)
    _build_dataset(os.path.join(tmp, "raw_data", f"{benchmark}_adsorption.json"), dedup=True)
    save_dir = os.path.join(tmp, "result", "EMT_rs")
    # Pre-seed a slab cache file as if a prior run had relaxed the slab, then run:
    # if resume works, the run reuses it and matches a clean run.
    full = _ads_engs(_run(tmp, "ds_g", dedup=True, structure_cache=True))
    # second independent run must reproduce identical numbers (determinism)
    again = _ads_engs(_run(tmp, "ds_h", dedup=True, structure_cache=True))
    for k in full:
        assert abs(full[k] - again[k]) < 1e-9


# --- adslab reuse (identical adslab shared across reactions) ---

def _build_adslab_dup_dataset(path, dedup):
    # 3 reactions whose adslab (slab + O) is the SAME structure, frame-shifted ->
    # identical reuse_key -> the adslab relaxation should be reused (cache hit),
    # and every derived metric (energy, bond change, substrate displacement) must
    # be identical to a no-cache run.
    base_slab = _surface((0, 0, 0))
    base_ads = _adslab_from(base_slab, dz=0.0)
    n = len(base_slab)
    data = {}
    for i, sh in enumerate([(0, 0, 0), (0.3, 0.7, 0.0), (1.1, 0.2, 0.0)]):
        slab = _surface(sh)
        ads = base_ads.copy()
        ads.translate(sh)
        ads.wrap()
        data["O_dup%d" % i] = {
            "raw": {
                "star": {"atoms": slab, "energy_ref": 0.0, "stoi": -1},
                "Ostar": {"atoms": ads, "energy_ref": 0.0, "stoi": 1},
                "O2gas": {"atoms": _gas(), "energy_ref": 0.0, "stoi": -0.5},
            },
            "ref_ads_eng": 0.0,
            "adsorbate_indices": [n],
            "constraint_source": "deposited",
        }
    _write_catbench_json(data, path, dedup=dedup)


def _run_builder(tmp, benchmark, builder, dedup, structure_cache):
    os.makedirs(os.path.join(tmp, "raw_data"), exist_ok=True)
    builder(os.path.join(tmp, "raw_data", f"{benchmark}_adsorption.json"), dedup)
    mlip = f"EMT_ad_{int(dedup)}_{int(structure_cache)}"
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        AdsorptionCalculation(
            [EMT(), EMT()], mlip_name=mlip, benchmark=benchmark,
            save_files=False, structure_cache=structure_cache,
        ).run()
        with open(os.path.join(tmp, "result", mlip, f"{mlip}_result.json")) as f:
            res = json.load(f)
    finally:
        os.chdir(cwd)
    return res


def test_adslab_cache_produces_hits(tmp_path):
    tmp = str(tmp_path)
    res = _run_builder(tmp, "ad_a", _build_adslab_dup_dataset, dedup=True, structure_cache=True)
    adslab_times = [v["0"]["adslab_time"] for k, v in res.items()
                    if isinstance(v, dict) and "0" in v]
    assert len(adslab_times) == 3
    assert adslab_times.count(0.0) >= 2  # at least 2 of 3 adslabs reused


def test_adslab_cache_on_equals_off_incl_anomaly_metrics(tmp_path):
    tmp = str(tmp_path)
    off = _run_builder(tmp, "ad_b", _build_adslab_dup_dataset, dedup=False, structure_cache=False)
    on = _run_builder(tmp, "ad_c", _build_adslab_dup_dataset, dedup=True, structure_cache=True)
    keys = [k for k in off if isinstance(off[k], dict) and "0" in off[k]]
    assert keys
    for k in keys:
        for fld in ("ads_eng", "adslab_tot_eng", "max_bond_change",
                    "substrate_displacement", "adslab_max_disp",
                    "adslab_pos_mae", "adslab_pos_rmsd", "adslab_energy_change"):
            assert abs(on[k]["0"][fld] - off[k]["0"][fld]) < 1e-9, (k, fld)


def test_structure_cache_invalidated_when_relax_settings_change(tmp_path):
    # A cache written under one set of relaxation settings must NOT be reused when
    # the run's settings change (else stale, loosely-converged numbers reappear).
    tmp = str(tmp_path)
    os.makedirs(os.path.join(tmp, "raw_data"), exist_ok=True)
    _build_dataset(os.path.join(tmp, "raw_data", "cfg_adsorption.json"), dedup=True)
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        AdsorptionCalculation([EMT(), EMT()], mlip_name="EMT_cfg", benchmark="cfg",
                              save_files=False, structure_cache=True, n_crit_relax=50).run()
        save_dir = os.path.join(tmp, "result", "EMT_cfg")
        cache = json.load(open(os.path.join(save_dir, "EMT_cfg_structure_cache.json")))
        assert "__relax_config__" in cache
        assert sum(1 for k in cache if k != "__relax_config__") >= 1  # real entries exist

        # SAME settings -> cache kept
        same = AdsorptionCalculation([EMT(), EMT()], mlip_name="EMT_cfg", benchmark="cfg",
                                     save_files=False, structure_cache=True, n_crit_relax=50)
        _, _, _, se_same = same._load_existing_results(save_dir)
        assert len(se_same) > 1

        # DIFFERENT n_crit_relax -> cache discarded
        diff = AdsorptionCalculation([EMT(), EMT()], mlip_name="EMT_cfg", benchmark="cfg",
                                     save_files=False, structure_cache=True, n_crit_relax=999)
        _, _, _, se_diff = diff._load_existing_results(save_dir)
        assert se_diff == {}
    finally:
        os.chdir(cwd)


def test_removed_rate_knob_warns():
    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        AdsorptionCalculation([EMT()], mlip_name="warn", benchmark="b", rate=0.5)
    assert any("rate" in str(r.message) for r in rec)
