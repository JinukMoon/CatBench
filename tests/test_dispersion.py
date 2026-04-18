import importlib


def test_dispersion_module_importable():
    module = importlib.import_module("catbench.dispersion")
    assert hasattr(module, "DispersionCorrection")
    assert hasattr(module, "DISPERSION_CONFIGS")
    assert hasattr(module, "get_dispersion_config")


def test_dispersion_configs_is_dict():
    from catbench.dispersion import DISPERSION_CONFIGS

    assert isinstance(DISPERSION_CONFIGS, dict)


def test_stub_raises_when_torch_absent():
    try:
        import torch  # noqa: F401
    except ImportError:
        from catbench.dispersion import DispersionCorrection

        import pytest

        with pytest.raises(ImportError):
            DispersionCorrection()
