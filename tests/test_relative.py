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


def test_custom_target_written_with_its_own_forces(empty_custom_json, tmp_path):
    """1.1.7: the shared calculator's results used to belong to the LAST reference when the
    target was written -> broadcast error whenever reference and target sizes differ."""
    from ase.build import bulk, molecule
    from ase.io import read
    calc = RelativeEnergyCalculation(EMT(), mlip_name="x", benchmark="demo", task_type="custom")
    target = bulk("Cu", "fcc", a=3.6, cubic=True)                     # 4 atoms
    o2 = molecule("O2"); o2.center(vacuum=5.0)                          # 2 atoms
    data = {"target": {"atoms": target, "energy": -1.0},
            "references": {"Cu": {"atoms": bulk("Cu", "fcc", a=3.6), "energy": -0.2},   # 1 atom
                           "O2": {"atoms": o2, "energy": -0.5}},
            "stoichiometry": {"target": 1, "Cu": -4, "O2": 0}, "normalization_factor": 1}
    res = calc._calculate_formation_energy("sys", data, str(tmp_path))
    t = read(tmp_path / "sys_target.extxyz")
    assert len(t) == 4 and t.get_forces().shape == (4, 3)
    assert t.get_potential_energy() == pytest.approx(res["mlip_calculation"]["target_energy"])
    assert len(read(tmp_path / "sys_O2.extxyz")) == 2
