import os

import pytest

from catbench.config import (
    ANALYSIS_DEFAULTS,
    CALCULATION_DEFAULTS,
    PLOT_COLORS,
    PLOT_MARKERS,
    RELATIVE_ANALYSIS_DEFAULTS,
    get_default,
)


REQUIRED_CALCULATION_KEYS = (
    "optimizer",
    "save_step",
    "save_files",
    "f_crit_relax",
    "n_crit_relax",
    "rate",
    "damping",
    "chemical_bond_cutoff",
)


REQUIRED_ANALYSIS_KEYS = (
    "calculating_path",
    "figsize",
    "dpi",
    "disp_thrs",
    "energy_thrs",
    "reproduction_thrs",
    "bond_length_change_threshold",
    "plot_enabled",
)


@pytest.mark.parametrize("key", REQUIRED_CALCULATION_KEYS)
def test_calculation_defaults_has_key(key):
    assert key in CALCULATION_DEFAULTS


@pytest.mark.parametrize("key", REQUIRED_ANALYSIS_KEYS)
def test_analysis_defaults_has_key(key):
    assert key in ANALYSIS_DEFAULTS


def test_relative_analysis_defaults_has_task_type_slot():
    assert "task_type" in RELATIVE_ANALYSIS_DEFAULTS


def test_get_default_passthrough_value():
    assert get_default("optimizer", CALCULATION_DEFAULTS) == "LBFGS"
    assert get_default("figsize", ANALYSIS_DEFAULTS) == (9, 8)


def test_get_default_resolves_lambda(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = get_default("calculating_path", ANALYSIS_DEFAULTS)
    assert resolved == os.path.join(str(tmp_path), "result")


def test_plot_color_and_marker_lists_are_nonempty():
    assert len(PLOT_COLORS) > 0
    assert len(PLOT_MARKERS) > 0
