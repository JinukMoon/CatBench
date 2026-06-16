"""Tests pinning the shared reaction classifier (C3) and NumpyEncoder."""

import json

import numpy as np
import pytest

from catbench.utils.analysis_utils import classify_reaction
from catbench.utils.calculation_utils import NumpyEncoder


# --------------------------------------------------------------------------- #
# classify_reaction priority order (C3)
# --------------------------------------------------------------------------- #
def test_classify_all_false_is_normal():
    assert classify_reaction(False, False, False, False) == "normal"


def test_classify_seed_has_highest_priority():
    # seed wins over everything else, regardless of other flags.
    assert (
        classify_reaction(True, True, True, True) == "reproduction_failure"
    )
    assert classify_reaction(True, False, False, False) == "reproduction_failure"


def test_classify_unphysical_beats_migration_and_energy():
    assert (
        classify_reaction(False, True, True, True) == "unphysical_relaxation"
    )


def test_classify_migration_only_when_not_unphysical():
    # migration is gated: it only fires when unphysical is False.
    assert classify_reaction(False, True, True, False) == "unphysical_relaxation"
    assert classify_reaction(False, False, True, True) == "adsorbate_migration"


def test_classify_energy_lowest_nonnormal():
    assert classify_reaction(False, False, False, True) == "energy_anomaly"


@pytest.mark.parametrize(
    "seed,unphysical,migration,energy,expected",
    [
        (True, True, True, True, "reproduction_failure"),
        (False, True, True, True, "unphysical_relaxation"),
        (False, False, True, True, "adsorbate_migration"),
        (False, False, False, True, "energy_anomaly"),
        (False, False, False, False, "normal"),
        (False, True, False, False, "unphysical_relaxation"),
    ],
)
def test_classify_full_priority_matrix(seed, unphysical, migration, energy, expected):
    assert classify_reaction(seed, unphysical, migration, energy) == expected


# --------------------------------------------------------------------------- #
# NumpyEncoder
# --------------------------------------------------------------------------- #
def test_numpy_encoder_handles_numpy_scalar_types():
    payload = {
        "b": np.bool_(True),
        "i": np.int64(7),
        "f": np.float64(1.5),
        "arr": np.array([1, 2, 3]),
    }
    encoded = json.dumps(payload, cls=NumpyEncoder)
    decoded = json.loads(encoded)

    assert decoded["b"] is True
    assert decoded["i"] == 7
    assert decoded["f"] == pytest.approx(1.5)
    assert decoded["arr"] == [1, 2, 3]


def test_numpy_encoder_handles_2d_ndarray():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    decoded = json.loads(json.dumps({"m": arr}, cls=NumpyEncoder))
    assert decoded["m"] == [[1.0, 2.0], [3.0, 4.0]]
