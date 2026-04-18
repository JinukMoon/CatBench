import pytest
from ase.calculators.emt import EMT

from catbench.adsorption import AdsorptionCalculation


def _dummy_calc():
    return EMT()


def test_rejects_non_list_calculator():
    with pytest.raises(ValueError):
        AdsorptionCalculation(_dummy_calc(), mlip_name="x", benchmark="y")


def test_rejects_empty_calculator_list():
    with pytest.raises(ValueError):
        AdsorptionCalculation([], mlip_name="x", benchmark="y")


def test_rejects_invalid_mode():
    with pytest.raises(ValueError):
        AdsorptionCalculation(
            [_dummy_calc()], mode="no_such_mode", mlip_name="x", benchmark="y"
        )


def test_requires_mlip_name():
    with pytest.raises(ValueError):
        AdsorptionCalculation([_dummy_calc()], benchmark="y")


def test_requires_benchmark():
    with pytest.raises(ValueError):
        AdsorptionCalculation([_dummy_calc()], mlip_name="x")


def test_init_applies_calculation_defaults():
    calc = AdsorptionCalculation([_dummy_calc()], mlip_name="x", benchmark="y")
    assert calc.config["optimizer"] == "LBFGS"
    assert calc.config["rate"] == 0.5
    assert calc.config["f_crit_relax"] == 0.05
    assert calc.config["save_files"] is True


def test_init_respects_user_overrides():
    calc = AdsorptionCalculation(
        [_dummy_calc()],
        mlip_name="x",
        benchmark="y",
        rate=None,
        save_files=False,
        f_crit_relax=0.01,
    )
    assert calc.config["rate"] is None
    assert calc.config["save_files"] is False
    assert calc.config["f_crit_relax"] == 0.01
    assert calc.config["optimizer"] == "LBFGS"


def test_oc20_mode_accepted():
    calc = AdsorptionCalculation(
        [_dummy_calc()], mode="oc20", mlip_name="x", benchmark="y"
    )
    assert calc.mode == "oc20"
