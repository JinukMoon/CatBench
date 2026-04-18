import pytest
from ase.calculators.emt import EMT

from catbench.relative import RelativeEnergyCalculation


def test_rejects_invalid_task_type():
    with pytest.raises(ValueError):
        RelativeEnergyCalculation(
            EMT(), mlip_name="x", benchmark="demo", task_type="not_a_task"
        )


def test_requires_mlip_name(empty_surface_json):
    with pytest.raises(ValueError):
        RelativeEnergyCalculation(EMT(), benchmark="demo", task_type="surface")


def test_requires_benchmark(empty_surface_json):
    with pytest.raises(ValueError):
        RelativeEnergyCalculation(EMT(), mlip_name="x", task_type="surface")


def test_requires_task_type(empty_surface_json):
    with pytest.raises(ValueError):
        RelativeEnergyCalculation(EMT(), mlip_name="x", benchmark="demo")


def test_missing_data_file_raises(chdir_tmp):
    with pytest.raises(FileNotFoundError):
        RelativeEnergyCalculation(
            EMT(), mlip_name="x", benchmark="missing", task_type="surface"
        )


def test_init_surface(empty_surface_json):
    calc = RelativeEnergyCalculation(
        EMT(), mlip_name="x", benchmark="demo", task_type="surface"
    )
    assert calc.task_type == "surface"
    assert calc.data == {}


def test_init_bulk_formation(empty_bulk_formation_json):
    calc = RelativeEnergyCalculation(
        EMT(), mlip_name="x", benchmark="demo", task_type="bulk_formation"
    )
    assert calc.task_type == "bulk_formation"


def test_init_custom(empty_custom_json):
    calc = RelativeEnergyCalculation(
        EMT(), mlip_name="x", benchmark="demo", task_type="custom"
    )
    assert calc.task_type == "custom"
