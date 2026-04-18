import hashlib

import pytest

from catbench.adsorption import list_zenodo_benchmarks, zenodo_download
from catbench.adsorption.data import zenodo as zmod


def test_list_benchmarks_returns_five():
    names = list_zenodo_benchmarks()
    assert set(names) == {
        "MamunHighT2019",
        "FG_dataset",
        "KHLOHC_origin",
        "ComerGeneralized2024",
        "BM_dataset",
    }


def test_rejects_unknown_benchmark(chdir_tmp):
    with pytest.raises(ValueError, match="Unknown benchmark"):
        zenodo_download("does_not_exist")


def test_error_message_lists_available(chdir_tmp):
    with pytest.raises(ValueError, match="MamunHighT2019"):
        zenodo_download("typo")


def test_skips_if_file_already_exists(chdir_tmp, capsys):
    raw = chdir_tmp / "raw_data"
    raw.mkdir()
    existing = raw / "BM_dataset_adsorption.json"
    existing.write_text("already downloaded")

    result = zenodo_download("BM_dataset")

    assert result == str(existing)
    assert existing.read_text() == "already downloaded"
    assert "Already exists" in capsys.readouterr().out


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise zmod.requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        for i in range(0, len(self._payload), chunk_size):
            yield self._payload[i : i + chunk_size]


def _patch_requests(monkeypatch, payload):
    def fake_get(url, **kwargs):
        return _FakeResponse(payload)

    monkeypatch.setattr(zmod.requests, "get", fake_get)


def test_successful_download_writes_file(chdir_tmp, monkeypatch):
    payload = b'{"demo": true}'
    _patch_requests(monkeypatch, payload)

    result = zenodo_download("BM_dataset", verify=False)

    target = chdir_tmp / "raw_data" / "BM_dataset_adsorption.json"
    assert target.exists()
    assert target.read_bytes() == payload
    assert result == str(target)


def test_md5_mismatch_deletes_file(chdir_tmp, monkeypatch):
    _patch_requests(monkeypatch, b"not the real content")

    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        zenodo_download("BM_dataset")

    target = chdir_tmp / "raw_data" / "BM_dataset_adsorption.json"
    assert not target.exists()


def test_md5_verify_accepts_matching_content(chdir_tmp, monkeypatch):
    payload = b"synthetic-content-for-testing"
    real_md5 = hashlib.md5(payload).hexdigest()

    monkeypatch.setitem(
        zmod._ZENODO_BENCHMARKS,
        "BM_dataset",
        {
            "filename": "BM_dataset_adsorption.json",
            "md5": real_md5,
            "size": len(payload),
        },
    )
    _patch_requests(monkeypatch, payload)

    result = zenodo_download("BM_dataset")
    assert (chdir_tmp / "raw_data" / "BM_dataset_adsorption.json").exists()
    assert result.endswith("BM_dataset_adsorption.json")


def test_network_error_cleans_up_partial_file(chdir_tmp, monkeypatch):
    def failing_get(url, **kwargs):
        raise zmod.requests.ConnectionError("network down")

    monkeypatch.setattr(zmod.requests, "get", failing_get)

    with pytest.raises(RuntimeError, match="Download failed"):
        zenodo_download("BM_dataset")

    target = chdir_tmp / "raw_data" / "BM_dataset_adsorption.json"
    assert not target.exists()


def test_overwrite_flag_redownloads(chdir_tmp, monkeypatch):
    raw = chdir_tmp / "raw_data"
    raw.mkdir()
    target = raw / "BM_dataset_adsorption.json"
    target.write_text("stale content")

    payload = b'{"fresh": true}'
    _patch_requests(monkeypatch, payload)

    zenodo_download("BM_dataset", overwrite=True, verify=False)
    assert target.read_bytes() == payload
