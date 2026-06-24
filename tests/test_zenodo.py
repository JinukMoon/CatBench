import gzip
import hashlib

import pytest

from catbench.adsorption import get_benchmark, list_zenodo_benchmarks, zenodo_download
from catbench.adsorption.data import zenodo as zmod


def _fake_files(payload=b"x", md5=None):
    """A fake _zenodo_latest_files() result (no network)."""
    return {
        "BM_dataset": {
            "filename": "BM_dataset_adsorption.json",
            "url": "https://fake.zenodo/BM_dataset_adsorption.json/content",
            "md5": md5 if md5 is not None else hashlib.md5(payload).hexdigest(),
            "size": len(payload),
        },
        "MamunHighT2019": {
            "filename": "MamunHighT2019_adsorption.json",
            "url": "https://fake.zenodo/MamunHighT2019_adsorption.json/content",
            "md5": "deadbeef", "size": 100,
        },
    }


@pytest.fixture
def patch_latest(monkeypatch):
    def _apply(files):
        monkeypatch.setattr(zmod, "_zenodo_latest_files", lambda force=False: files)
    return _apply


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.content = payload
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


def _patch_get(monkeypatch, payload):
    monkeypatch.setattr(zmod.requests, "get", lambda url, **kw: _FakeResponse(payload))


# --- list / resolution (version-agnostic) ---

def test_list_benchmarks_is_dynamic(patch_latest):
    patch_latest(_fake_files())
    assert set(list_zenodo_benchmarks()) == {"BM_dataset", "MamunHighT2019"}


def test_rejects_unknown_benchmark(chdir_tmp, patch_latest):
    patch_latest(_fake_files())
    with pytest.raises(ValueError, match="not on the CatBench Zenodo record"):
        zenodo_download("does_not_exist")


def test_skips_if_file_already_exists(chdir_tmp, capsys):
    # exists check happens BEFORE any network/API call
    raw = chdir_tmp / "raw_data"
    raw.mkdir()
    existing = raw / "BM_dataset_adsorption.json"
    existing.write_text("already downloaded")
    result = zenodo_download("BM_dataset")
    assert result == str(existing)
    assert existing.read_text() == "already downloaded"
    assert "Already exists" in capsys.readouterr().out


# --- download mechanics ---

def test_successful_download_writes_file(chdir_tmp, patch_latest, monkeypatch):
    payload = b'{"demo": true}'
    patch_latest(_fake_files(payload))
    _patch_get(monkeypatch, payload)
    result = zenodo_download("BM_dataset", verify=False)
    target = chdir_tmp / "raw_data" / "BM_dataset_adsorption.json"
    assert target.exists() and target.read_bytes() == payload
    assert result == str(target)


def test_md5_verify_accepts_matching_content(chdir_tmp, patch_latest, monkeypatch):
    payload = b"synthetic-content-for-testing"
    patch_latest(_fake_files(payload))  # md5 derived from payload
    _patch_get(monkeypatch, payload)
    result = zenodo_download("BM_dataset")
    assert result.endswith("BM_dataset_adsorption.json")


def test_md5_mismatch_deletes_file(chdir_tmp, patch_latest, monkeypatch):
    patch_latest(_fake_files(b"real", md5="0" * 32))
    _patch_get(monkeypatch, b"corrupted-content")
    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        zenodo_download("BM_dataset")
    assert not (chdir_tmp / "raw_data" / "BM_dataset_adsorption.json").exists()


def test_network_error_cleans_up_partial_file(chdir_tmp, patch_latest, monkeypatch):
    patch_latest(_fake_files())

    def boom(url, **kw):
        raise zmod.requests.ConnectionError("network down")

    monkeypatch.setattr(zmod.requests, "get", boom)
    with pytest.raises(RuntimeError, match="Download failed"):
        zenodo_download("BM_dataset", verify=False)
    assert not (chdir_tmp / "raw_data" / "BM_dataset_adsorption.json").exists()


def test_overwrite_flag_redownloads(chdir_tmp, patch_latest, monkeypatch):
    raw = chdir_tmp / "raw_data"
    raw.mkdir()
    target = raw / "BM_dataset_adsorption.json"
    target.write_text("stale content")
    payload = b'{"fresh": true}'
    patch_latest(_fake_files(payload))
    _patch_get(monkeypatch, payload)
    zenodo_download("BM_dataset", overwrite=True, verify=False)
    assert target.read_bytes() == payload


# --- get_benchmark 3-tier routing ---

def test_get_benchmark_tier1_zenodo(chdir_tmp, patch_latest, monkeypatch):
    payload = b'{"tier": 1}'
    patch_latest(_fake_files(payload))
    _patch_get(monkeypatch, payload)
    get_benchmark("BM_dataset", verify=False)
    assert (chdir_tmp / "raw_data" / "BM_dataset_adsorption.json").read_bytes() == payload


def test_get_benchmark_tier2_leaderboard(chdir_tmp, patch_latest, monkeypatch):
    patch_latest({})  # not on Zenodo
    monkeypatch.setattr(zmod, "_leaderboard_has", lambda name: True)
    gz = gzip.compress(b'{"tier": 2}')
    monkeypatch.setattr(zmod.requests, "get", lambda url, **kw: _FakeResponse(gz))
    get_benchmark("SomeCatHubSet2024")
    target = chdir_tmp / "raw_data" / "SomeCatHubSet2024_adsorption.json"
    assert target.read_bytes() == b'{"tier": 2}'


def test_get_benchmark_tier3_cathub_fallback(chdir_tmp, patch_latest, monkeypatch):
    patch_latest({})  # not on Zenodo
    monkeypatch.setattr(zmod, "_leaderboard_has", lambda name: False)
    called = {}
    import catbench.adsorption.data.cathub as cathub_mod
    monkeypatch.setattr(cathub_mod, "cathub_preprocessing",
                        lambda name, **kw: called.setdefault("name", name))
    get_benchmark("BrandNewCatHub2027")
    assert called.get("name") == "BrandNewCatHub2027"
