"""Download pre-formatted CatBench benchmark datasets.

Three sources, resolved automatically by :func:`get_benchmark`:

1. **Zenodo** -- the featured/headline datasets, citable with a DOI. Resolved
   version-agnostically through the *concept* record, so new Zenodo versions are
   picked up automatically with no code change.
2. **CatBench leaderboard CDN** (``https://catbench.org/benchmark/<name>.json.gz``)
   -- every dataset shown on catbench.org, gzip-compressed (fast).
3. **CatHub** -- direct download + preprocessing for anything not hosted above.

Zenodo concept DOI: 10.5281/zenodo.17157085 (always resolves to the latest version).
"""

import gzip
import hashlib
import os
import sys

import requests


# The ONLY hardcoded id: the Zenodo *concept* record. It never changes across
# versions, and the API resolves it to the current latest version (+ its files,
# checksums, and download links), so publishing a new Zenodo version needs no
# code change here.
_ZENODO_CONCEPT_ID = "17157085"
_ZENODO_CONCEPT_DOI = "10.5281/zenodo.17157085"
_LEADERBOARD_BASE = "https://catbench.org/benchmark"
_SUFFIX = "_adsorption.json"

_latest_cache = None


def _zenodo_latest_files(force=False):
    """Resolve the concept record to the latest version's files.

    Returns ``{benchmark_name: {"filename", "url", "md5", "size"}}`` built live
    from the Zenodo API, so the set of available benchmarks and their checksums
    always reflect the current latest Zenodo version.
    """
    global _latest_cache
    if _latest_cache is not None and not force:
        return _latest_cache
    url = f"https://zenodo.org/api/records/{_ZENODO_CONCEPT_ID}"
    rec = requests.get(url, timeout=60)
    rec.raise_for_status()
    rec = rec.json()
    record_id = rec["id"]
    out = {}
    for f in rec.get("files", []):
        key = f.get("key", "")
        if not key.endswith(_SUFFIX):
            continue
        name = key[: -len(_SUFFIX)]
        checksum = f.get("checksum", "") or ""
        out[name] = {
            "filename": key,
            "url": f"https://zenodo.org/api/records/{record_id}/files/{key}/content",
            "md5": checksum.split(":")[-1] if checksum else None,
            "size": f.get("size"),
        }
    _latest_cache = out
    return out


def list_zenodo_benchmarks():
    """Return the benchmark names hosted on the latest CatBench Zenodo version."""
    return sorted(_zenodo_latest_files().keys())


def _target_path(name):
    target_dir = os.path.join(os.getcwd(), "raw_data")
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, f"{name}{_SUFFIX}")


def _stream_download(url, target_path, expected_size=None, expected_md5=None, label=""):
    md5 = hashlib.md5()
    downloaded = 0
    last_reported = 0
    chunk_size = 1024 * 1024
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    md5.update(chunk)
                    downloaded += len(chunk)
                    if expected_size and downloaded - last_reported >= 10 * chunk_size:
                        pct = 100 * downloaded / expected_size
                        sys.stdout.write(
                            f"\r  {downloaded / 1e6:>7.1f} / "
                            f"{expected_size / 1e6:.1f} MB ({pct:5.1f}%)"
                        )
                        sys.stdout.flush()
                        last_reported = downloaded
            sys.stdout.write("\n")
            sys.stdout.flush()
    except requests.RequestException as e:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise RuntimeError(f"Download failed for {label or url}: {e}") from e

    if expected_md5:
        actual = md5.hexdigest()
        if actual != expected_md5:
            os.remove(target_path)
            raise RuntimeError(
                f"MD5 mismatch for {label}: expected {expected_md5}, got {actual}. "
                "Corrupted file removed."
            )
        print(f"MD5 verified: {actual}")


def zenodo_download(benchmark, overwrite=False, verify=True):
    """Download a featured benchmark JSON from Zenodo into ``raw_data/``.

    Explicit Zenodo-only path. Use :func:`get_benchmark` for automatic
    Zenodo -> leaderboard -> CatHub routing.
    """
    target_path = _target_path(benchmark)
    if os.path.exists(target_path) and not overwrite:
        print(f"Already exists: {target_path}")
        print("Pass overwrite=True to redownload.")
        return target_path
    files = _zenodo_latest_files()
    if benchmark not in files:
        raise ValueError(
            f"'{benchmark}' is not on the CatBench Zenodo record. "
            f"Available: {sorted(files)}"
        )
    meta = files[benchmark]
    print(f"Downloading {benchmark} from Zenodo ({(meta['size'] or 0) / 1e6:.1f} MB)")
    print(f"  Target: {target_path}")
    _stream_download(meta["url"], target_path, meta["size"],
                     meta["md5"] if verify else None, label=benchmark)
    print(f"Saved: {target_path}")
    return target_path


def _leaderboard_url(name):
    return f"{_LEADERBOARD_BASE}/{name}.json.gz"


def _leaderboard_has(name):
    try:
        r = requests.head(_leaderboard_url(name), timeout=30, allow_redirects=True)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _leaderboard_download(name, target_path):
    url = _leaderboard_url(name)
    print(f"Downloading {name} from CatBench leaderboard (gzip)")
    print(f"  URL:    {url}")
    print(f"  Target: {target_path}")
    try:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        data = gzip.decompress(r.content)
    except (requests.RequestException, OSError) as e:
        raise RuntimeError(f"Leaderboard download failed for {name}: {e}") from e
    with open(target_path, "wb") as f:
        f.write(data)
    print(f"Saved: {target_path}")
    return target_path


def get_benchmark(name, overwrite=False, verify=True):
    """Fetch a benchmark dataset, trying each source in order of preference.

    1. Zenodo (featured, citable) -> 2. catbench.org leaderboard (gzip, fast) ->
    3. CatHub (direct preprocessing). The first source that has ``name`` wins, so
    callers do not need to know where a dataset lives.

    Returns the path to ``raw_data/<name>_adsorption.json``.
    """
    target_path = _target_path(name)
    if os.path.exists(target_path) and not overwrite:
        print(f"Already exists: {target_path}")
        return target_path

    # Tier 1: Zenodo (featured)
    try:
        zfiles = _zenodo_latest_files()
    except Exception:
        zfiles = {}
    if name in zfiles:
        print(f"[get_benchmark] '{name}' -> Zenodo")
        return zenodo_download(name, overwrite=overwrite, verify=verify)

    # Tier 2: leaderboard CDN
    if _leaderboard_has(name):
        print(f"[get_benchmark] '{name}' -> catbench.org leaderboard")
        return _leaderboard_download(name, target_path)

    # Tier 3: CatHub fallback
    print(f"[get_benchmark] '{name}' -> CatHub (fallback)")
    from catbench.adsorption.data.cathub import cathub_preprocessing
    cathub_preprocessing(name)
    return target_path
