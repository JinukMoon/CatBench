"""Regression: anomaly-sheet detail columns must not leak a reaction across groups.

A reaction can trip several raw anomaly flags at once (e.g. both a seed/
reproduction anomaly and an unphysical-relaxation anomaly), but classify_reaction
assigns it to a single exclusive parent bucket by priority. The detail breakdown
columns must follow that classification -- a reproduction_failure reaction must
NOT also be counted in the unphysical-relaxation detail columns.

This builds a reaction with BOTH a seed flag and an unphysical flag, and asserts
the unphysical detail column equals the unphysical parent count (no leak).
"""
import json
import os

import pytest

from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from catbench.utils.data_utils import save_catbench_json
from catbench.adsorption import AdsorptionCalculation, AdsorptionAnalysis

# Anomaly sheet column order (row after the 2 header rows).
_COLS = ["MLIP", "Num_normal", "Num_migration", "anom_total", "repro_parent",
         "unphys_parent", "energy", "ads_eng_seed", "ads_seed", "slab_seed",
         "ads_conv", "ads_move", "slab_conv", "slab_move"]


def _build_and_run(tmp):
    os.makedirs(os.path.join(tmp, "raw_data"), exist_ok=True)
    data = {}
    for i in range(3):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
        slab.set_constraint(
            FixAtoms(mask=[a.position[2] < slab.positions[:, 2].mean() for a in slab])
        )
        ads = slab.copy()
        tz = ads.positions[:, 2].max()
        ads += Atoms("O", positions=[[ads.cell[0, 0] / 2, ads.cell[1, 1] / 2, tz + 1.7 + 0.1 * i]])
        n = len(slab)
        data["O_%d" % i] = {
            "raw": {"star": {"atoms": slab, "energy_ref": -1.0, "stoi": -1},
                    "Ostar": {"atoms": ads, "energy_ref": 0.0, "stoi": 1}},
            "ref_ads_eng": 0.0, "adsorbate_indices": [n], "constraint_source": "deposited",
        }
    save_catbench_json(data, os.path.join(tmp, "raw_data", "AN_adsorption.json"))
    AdsorptionCalculation([EMT(), EMT(), EMT()], mlip_name="EMT", benchmark="AN",
                          save_files=False, n_crit_relax=80).run()


def _anomaly_row(tmp, openpyxl):
    xlsx = [f for f in os.listdir(tmp) if f.endswith("_Benchmarking_Analysis.xlsx")][0]
    wb = openpyxl.load_workbook(os.path.join(tmp, xlsx), data_only=True)
    rows = [list(r) for r in wb["anomaly"].iter_rows(values_only=True)]
    data_rows = [r for r in rows if r and r[0] == "EMT"]
    return dict(zip(_COLS, data_rows[0]))


def test_anomaly_detail_columns_do_not_leak_across_groups(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    tmp = str(tmp_path)
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        _build_and_run(tmp)
        res_path = os.path.join(tmp, "result", "EMT", "EMT_result.json")
        r = json.load(open(res_path))
        ncrit = r.get("calculation_settings", {}).get("n_crit_relax", 80)

        def set_steps(rx, v):
            for s in ("0", "1", "2"):
                r[rx][s]["adslab_steps"] = v

        # O_0: seed flag (ads_eng_seed_range > reproduction_thrs) AND unphysical
        #      flag (adslab_steps == n_crit) -> classified reproduction_failure.
        r["O_0"]["final"]["ads_eng_seed_range"] = 0.5
        r["O_0"]["final"]["slab_seed_range"] = 0.0
        r["O_0"]["final"]["ads_seed_range"] = 0.0
        set_steps("O_0", ncrit)
        # O_1: unphysical flag only -> classified unphysical_relaxation.
        for k in ("ads_eng_seed_range", "slab_seed_range", "ads_seed_range"):
            r["O_1"]["final"][k] = 0.0
        set_steps("O_1", ncrit)
        json.dump(r, open(res_path, "w"))

        AdsorptionAnalysis(plot_enabled=False).analysis()
        d = _anomaly_row(tmp, openpyxl)

        # O_0 -> reproduction_failure, O_1 -> unphysical_relaxation
        assert d["repro_parent"] == 1, d
        assert d["unphys_parent"] == 1, d
        # O_0 counted in its OWN group's detail column
        assert d["ads_eng_seed"] == 1, d
        # KEY: the unphysical detail column must not leak the repro-classified O_0.
        # ads_conv must equal the unphysical parent (only O_1), i.e. 1 -- not 2.
        assert d["ads_conv"] == d["unphys_parent"], (
            f"anomaly detail column leaked across groups: "
            f"ads_conv={d['ads_conv']} != unphys_parent={d['unphys_parent']} ({d})"
        )
    finally:
        os.chdir(cwd)
