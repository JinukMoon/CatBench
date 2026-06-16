"""Tests pinning NaN-safe median selection (A2), failure surfacing (A1), and
the matplotlib-free lazy import contract (B1)."""

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from catbench.utils.calculation_utils import find_median_index
from catbench.utils.io_utils import (
    load_existing_results,
    save_calculation_results,
)


# --------------------------------------------------------------------------- #
# find_median_index NaN-safe (A2)
# --------------------------------------------------------------------------- #
def test_find_median_index_basic():
    # sorted ranks: 1.0, 3.0, 5.0 -> median rank index points to value 3.0
    idx, val = find_median_index([5.0, 1.0, 3.0])
    assert idx == 2
    assert val == pytest.approx(3.0)


def test_find_median_index_nan_does_not_return_none_or_crash():
    idx, val = find_median_index([1.0, float("nan"), 3.0])
    # NaN sorts to the end, so the median-rank element is a real value (3.0),
    # never None and never a crash.
    assert idx is not None
    assert not math.isnan(val)
    assert val == pytest.approx(3.0)


def test_find_median_index_empty_raises_value_error():
    with pytest.raises(ValueError):
        find_median_index([])


def test_find_median_index_deterministic_on_ties():
    arr = [2.0, 2.0, 2.0, 2.0, 2.0]
    first = find_median_index(arr)
    for _ in range(5):
        assert find_median_index(arr) == first
    # median rank of 5 elements -> index (5-1)//2 = 2
    assert first[0] == 2


def test_find_median_index_returns_original_index():
    # Largest value first; the median-rank element must map back to its original
    # position in the input, not the sorted position.
    arr = [9.0, 1.0, 5.0]
    idx, val = find_median_index(arr)
    assert arr[idx] == pytest.approx(val)
    assert val == pytest.approx(5.0)
    assert idx == 2


# --------------------------------------------------------------------------- #
# A1 failure surfacing (offline, narrow contract on result-reading path)
# --------------------------------------------------------------------------- #
def test_failures_recorded_but_excluded_from_completed_reactions(tmp_path):
    """save_calculation_results writes _failures into the result file, but
    load_existing_results must NOT treat _failures or calculation_settings as
    completed reactions on resume."""
    mlip = "dummy"
    save_dir = tmp_path

    result_data = {
        "rxn_0": {"reference": {"ads_eng": -1.0}},
        "rxn_1": {"reference": {"ads_eng": -2.0}},
    }
    failures = {
        "rxn_bad": {"error": "RuntimeError('boom')", "traceback": "..."},
    }
    settings = {"optimizer": "LBFGS", "rate": None, "constraint_mode": "fixatoms"}

    save_calculation_results(
        str(save_dir),
        mlip,
        result_data,
        calculation_settings=settings,
        failures=failures,
    )

    # The on-disk file carries _failures and calculation_settings.
    with open(save_dir / f"{mlip}_result.json") as f:
        raw = json.load(f)
    assert "_failures" in raw
    assert "rxn_bad" in raw["_failures"]
    assert "calculation_settings" in raw

    # But the resume path returns only real reactions.
    final_result, _gas, _gas_single = load_existing_results(str(save_dir), mlip)
    assert set(final_result.keys()) == {"rxn_0", "rxn_1"}
    assert "_failures" not in final_result
    assert "calculation_settings" not in final_result
    # The failed reaction is NOT counted as completed -> it would be retried.
    assert "rxn_bad" not in final_result


def test_dummy_calculator_failure_does_not_abort_run(tmp_path, monkeypatch):
    """Drive the basic run loop's exception handling with a calculator that
    raises, asserting the run completes and records the failure under _failures
    instead of aborting.

    We use a minimal in-memory reaction whose single-point step calls the
    calculator immediately, so the failure path is exercised without any
    network/GPU and without a full relaxation.
    """
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from catbench.adsorption import AdsorptionCalculation

    class ExplodingCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            raise RuntimeError("calculator boom")

    def _atoms():
        return Atoms(
            "Cu2",
            positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            cell=[20.0, 20.0, 20.0],
            pbc=True,
        )

    # One reaction with a non-gas structure so the single-point call fires early.
    ref_data = {
        "rxn_0": {
            "ref_ads_eng": -1.0,
            "adsorbate_indices": [1],
            "raw": {
                "star": {"atoms": _atoms(), "energy_ref": -1.0, "stoi": -1},
                "OH": {"atoms": _atoms(), "energy_ref": -2.0, "stoi": 1},
            },
        }
    }

    calc = AdsorptionCalculation(
        [ExplodingCalculator()],
        mlip_name="dummy",
        benchmark="y",
        save_files=False,
    )

    save_dir = str(tmp_path / "result")
    monkeypatch.setattr(calc, "_load_data", lambda: ref_data)
    monkeypatch.setattr(calc, "_setup_directories", lambda: save_dir)

    import os

    os.makedirs(save_dir, exist_ok=True)

    # Must NOT raise: the failing reaction is caught and recorded.
    returned = calc._run_basic()
    assert returned == save_dir

    with open(os.path.join(save_dir, "dummy_result.json")) as f:
        raw = json.load(f)
    assert "_failures" in raw
    assert "rxn_0" in raw["_failures"]

    # On resume, the failed reaction is not treated as completed.
    final_result, _g, _gs = load_existing_results(save_dir, "dummy")
    assert "rxn_0" not in final_result


# --------------------------------------------------------------------------- #
# B1 lazy import: importing catbench.adsorption must not import matplotlib
# --------------------------------------------------------------------------- #
def test_import_adsorption_does_not_load_matplotlib():
    code = (
        "import sys\n"
        "from catbench.adsorption import AdsorptionCalculation\n"
        "assert AdsorptionCalculation is not None\n"
        "print('matplotlib' in sys.modules)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False", proc.stdout


def test_accessing_analysis_attribute_makes_it_importable():
    code = (
        "import catbench.adsorption as ads\n"
        "cls = ads.AdsorptionAnalysis\n"
        "assert cls is not None\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok", proc.stdout
