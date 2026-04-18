def test_top_level_package():
    import catbench

    assert hasattr(catbench, "__version__")
    assert isinstance(catbench.__version__, str)


def test_adsorption_public_api():
    from catbench.adsorption import (
        AdsorptionAnalysis,
        AdsorptionCalculation,
        cathub_preprocessing,
        process_output,
        vasp_preprocessing,
    )

    assert AdsorptionCalculation is not None
    assert AdsorptionAnalysis is not None
    assert callable(cathub_preprocessing)
    assert callable(vasp_preprocessing)
    assert callable(process_output)


def test_relative_public_api():
    from catbench.relative import (
        BulkFormationCalculation,
        RelativeEnergyAnalysis,
        RelativeEnergyCalculation,
        SurfaceEnergyCalculation,
    )

    assert RelativeEnergyCalculation is not None
    assert RelativeEnergyAnalysis is not None
    assert SurfaceEnergyCalculation is not None
    assert BulkFormationCalculation is not None


def test_eos_public_api():
    from catbench.eos import EOSAnalysis, EOSCalculation, eos_vasp_preprocessing

    assert EOSCalculation is not None
    assert EOSAnalysis is not None
    assert callable(eos_vasp_preprocessing)


def test_dispersion_import_never_crashes():
    from catbench.dispersion import DispersionCorrection

    assert DispersionCorrection is not None


def test_utils_submodules():
    from catbench.utils import analysis_utils, calculation_utils, data_utils, io_utils

    for module in (analysis_utils, calculation_utils, data_utils, io_utils):
        assert module is not None
