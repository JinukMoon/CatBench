import pytest
from ase.calculators.emt import EMT

from catbench.eos import EOSCalculation


def test_missing_data_file_raises(chdir_tmp):
    with pytest.raises(FileNotFoundError):
        EOSCalculation(calculator=EMT(), mlip_name="x", benchmark="missing")


def test_init_with_empty_dataset(empty_eos_json):
    calc = EOSCalculation(calculator=EMT(), mlip_name="x", benchmark="demo")
    assert calc.mlip_name == "x"
    assert calc.benchmark == "demo"
    assert calc.data == {}
