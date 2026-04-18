"""Direct download of pre-formatted CatBench benchmark datasets from Zenodo.

Zenodo record: https://zenodo.org/records/17157086
DOI: 10.5281/zenodo.17157086
"""

import hashlib
import os
import sys

import requests


_ZENODO_RECORD_ID = "17157086"
_ZENODO_DOI = "10.5281/zenodo.17157086"

_ZENODO_BENCHMARKS = {
    "MamunHighT2019": {
        "filename": "MamunHighT2019_adsorption.json",
        "md5": "8a208683009546b4d0f9821113ab1cc6",
        "size": 195405896,
    },
    "FG_dataset": {
        "filename": "FG_dataset_adsorption.json",
        "md5": "f7c22d01325e4171056aa60e73e1eca9",
        "size": 15345167,
    },
    "KHLOHC_origin": {
        "filename": "KHLOHC_origin_adsorption.json",
        "md5": "9e98a45f36d15604dbc8e9dfe9c2f0b5",
        "size": 10685170,
    },
    "ComerGeneralized2024": {
        "filename": "ComerGeneralized2024_adsorption.json",
        "md5": "0b6ad7380c19ab8d4b8d7cb016487f2e",
        "size": 1851867,
    },
    "BM_dataset": {
        "filename": "BM_dataset_adsorption.json",
        "md5": "ef2d57ef9f92047dec62c3222c33aeb6",
        "size": 445546,
    },
}


def list_zenodo_benchmarks():
    """Return the benchmark names available on the CatBench Zenodo record."""
    return sorted(_ZENODO_BENCHMARKS.keys())


def zenodo_download(benchmark, overwrite=False, verify=True):
    """Download a pre-formatted benchmark JSON from Zenodo into raw_data/.

    Args:
        benchmark: One of the names from ``list_zenodo_benchmarks()``.
        overwrite: If False, skip download when target already exists.
        verify: If True, verify the downloaded file's MD5 against Zenodo's
            published checksum and delete the file on mismatch.

    Returns:
        Absolute path to the downloaded (or already-present) JSON file.
    """
    if benchmark not in _ZENODO_BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. Available: "
            f"{list_zenodo_benchmarks()}"
        )

    meta = _ZENODO_BENCHMARKS[benchmark]
    filename = meta["filename"]
    expected_md5 = meta["md5"]
    expected_size = meta["size"]

    target_dir = os.path.join(os.getcwd(), "raw_data")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)

    if os.path.exists(target_path) and not overwrite:
        print(f"Already exists: {target_path}")
        print("Pass overwrite=True to redownload.")
        return target_path

    url = (
        f"https://zenodo.org/api/records/{_ZENODO_RECORD_ID}"
        f"/files/{filename}/content"
    )

    print(
        f"Downloading {benchmark} from Zenodo "
        f"({expected_size / 1e6:.1f} MB)"
    )
    print(f"  URL:    {url}")
    print(f"  Target: {target_path}")

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
                    if downloaded - last_reported >= 10 * chunk_size:
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
        raise RuntimeError(f"Download failed for {benchmark}: {e}") from e

    if verify:
        actual_md5 = md5.hexdigest()
        if actual_md5 != expected_md5:
            os.remove(target_path)
            raise RuntimeError(
                f"MD5 mismatch for {benchmark}: expected {expected_md5}, "
                f"got {actual_md5}. Corrupted file removed."
            )
        print(f"MD5 verified: {actual_md5}")

    print(f"Saved: {target_path}")
    return target_path
