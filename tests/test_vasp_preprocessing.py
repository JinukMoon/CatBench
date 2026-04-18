import pytest

from catbench.adsorption.data.vasp import _validate_vasp_inputs


def _valid_coeff():
    return {
        "H": {"slab": -1, "adslab": 1, "H2gas": -0.5},
        "OH": {"slab": -1, "adslab": 1, "H2gas": 0.5, "H2Ogas": -1},
    }


def test_rejects_missing_dataset_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        _validate_vasp_inputs(str(tmp_path / "nope"), _valid_coeff())


def test_rejects_non_string_dataset_name(tmp_path):
    with pytest.raises(TypeError):
        _validate_vasp_inputs(123, _valid_coeff())


def test_rejects_empty_coeff(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        _validate_vasp_inputs(str(tmp_path), {})


def test_rejects_non_dict_adsorbate_entry(tmp_path):
    with pytest.raises(ValueError, match="must be a dict"):
        _validate_vasp_inputs(str(tmp_path), {"H": [1, 2, 3]})


def test_rejects_missing_slab_key(tmp_path):
    with pytest.raises(ValueError, match="slab"):
        _validate_vasp_inputs(
            str(tmp_path), {"H": {"adslab": 1, "H2gas": -0.5}}
        )


def test_rejects_missing_adslab_key(tmp_path):
    with pytest.raises(ValueError, match="adslab"):
        _validate_vasp_inputs(
            str(tmp_path), {"H": {"slab": -1, "H2gas": -0.5}}
        )


def test_rejects_gas_key_without_gas_suffix(tmp_path):
    with pytest.raises(ValueError, match="gas"):
        _validate_vasp_inputs(
            str(tmp_path), {"H": {"slab": -1, "adslab": 1, "H2": -0.5}}
        )


def test_accepts_valid_inputs(tmp_path):
    _validate_vasp_inputs(str(tmp_path), _valid_coeff())
