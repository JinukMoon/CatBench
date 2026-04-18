import json

import pytest


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def empty_raw_data(chdir_tmp):
    raw = chdir_tmp / "raw_data"
    raw.mkdir()
    return raw


def _write_empty(raw_data_dir, filename):
    path = raw_data_dir / filename
    path.write_text(json.dumps({}))
    return path


@pytest.fixture
def empty_surface_json(empty_raw_data):
    return _write_empty(empty_raw_data, "demo_surface_energy.json")


@pytest.fixture
def empty_bulk_formation_json(empty_raw_data):
    return _write_empty(empty_raw_data, "demo_bulk_formation.json")


@pytest.fixture
def empty_custom_json(empty_raw_data):
    return _write_empty(empty_raw_data, "demo_custom.json")


@pytest.fixture
def empty_eos_json(empty_raw_data):
    return _write_empty(empty_raw_data, "demo_eos.json")
