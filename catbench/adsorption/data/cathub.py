"""
CatHub data processing for CatBench.

This module provides functions for downloading and processing catalysis reaction data 
from the CatHub database.
"""

import os
import time
import copy
import yaml
import json
import traceback
import requests
import io
import logging
from ase.io import read
from ase.constraints import FixAtoms
from catbench.utils.io_utils import get_raw_data_directory, get_raw_data_path, save_json

GRAPHQL = "https://api.catalysis-hub.org/graphql"

# Preprocessed CatHub datasets published by the CatBench team (same format that
# cathub_preprocessing() writes). Tried first so most users never need a CatHub
# API key and never touch CatHub's request limits. Override for testing/mirrors.
MIRROR_URL = os.environ.get("CATBENCH_MIRROR_URL", "https://catbench.org/benchmark")

CATHUB_KEY_URL = "https://api.catalysis-hub.org/auth/login"
CATHUB_MAX_PAGE = 200            # server-side per-request row cap
CATHUB_MIN_INTERVAL = 6.5        # s between requests (CatHub limit: 10/min)
CATHUB_DEFAULT_BUDGET = 300      # requests per call (CatHub suspends accounts >500/day)

# Key + request bookkeeping for the current download. Kept at module level so that
# fetch(query) keeps its one-argument signature (scripts monkeypatch it).
_api_key = None
_last_request = [0.0]
_request_count = [0]


class CatHubAuthError(RuntimeError):
    """CatHub rejected the request because of a missing/invalid API key."""


class CatHubBudgetError(RuntimeError):
    """A CatHub download would exceed the request budget (account-suspension guard)."""


class MirrorUnavailableError(RuntimeError):
    """The catbench.org mirror could not be reached (network), as opposed to 'not in mirror'."""


def _resolve_api_key(api_key=None):
    """Explicit argument > CATHUB_API_KEY environment variable. Never logged."""
    if api_key is not None:
        if not str(api_key).strip():
            raise CatHubAuthError("api_key was passed but is empty.")
        return str(api_key).strip()
    env = os.environ.get("CATHUB_API_KEY", "").strip()
    return env or None


def _missing_key_message(tag, n_requests=None):
    need = f" This dataset needs about {n_requests} requests." if n_requests else ""
    return (
        f"'{tag}' is not available from the catbench.org mirror, so it must be downloaded "
        f"from CatHub, which now requires an API key:\n"
        f"  1) Get a key at {CATHUB_KEY_URL}\n"
        f"  2) export CATHUB_API_KEY=<your key>   (or pass api_key=... to cathub_preprocessing)\n"
        f"Note: CatHub allows 10 requests/minute and suspends accounts above 500 requests/day.{need}"
    )


def fetch(query):
    """Fetch data from the CatHub GraphQL API (rate-limited, key sent as a header)."""
    wait = CATHUB_MIN_INTERVAL - (time.time() - _last_request[0])
    if _last_request[0] and wait > 0:
        time.sleep(wait)
    headers = {"X-API-Key": _api_key} if _api_key else {}
    try:
        # POST + header only: the key never appears in a URL, log line or exception text.
        # No redirects: never forward the key to an unexpected host.
        resp = requests.post(GRAPHQL, json={"query": query}, headers=headers,
                             timeout=120, allow_redirects=False)
    finally:
        _last_request[0] = time.time()
        _request_count[0] += 1
    if resp.status_code == 401:
        raise CatHubAuthError(
            "CatHub rejected the API key (401). It may be missing, mistyped or regenerated; "
            f"get a new one at {CATHUB_KEY_URL}.")
    if resp.status_code == 403:
        raise CatHubAuthError("CatHub refused the request (403). The account may be suspended "
                              "or not allowed; check the CatHub account page.")
    if resp.status_code == 429:
        raise CatHubBudgetError("CatHub rate limit hit (429). Stop and retry later; "
                                "repeated requests can get the account suspended.")
    if 300 <= resp.status_code < 400:
        raise RuntimeError(f"CatHub answered with a redirect ({resp.status_code}); refusing to follow it.")
    resp.raise_for_status()
    payload = resp.json()
    # CatHub returns {"data": null, "errors": [...]} on query errors. Surface a
    # clear message instead of an opaque KeyError/'NoneType' downstream.
    if payload.get("data") is None:
        errors = payload.get("errors")
        raise RuntimeError(f"CatHub GraphQL query returned no data; errors: {errors}")
    return payload["data"]


def _count_reactions(pub_id):
    """One small request: server-side totalCount for a publication."""
    data = fetch(f'{{ reactions(pubId: "{pub_id}", first: 1) {{ totalCount }} }}')
    return data["reactions"]["totalCount"]


def download_from_mirror(tag, dest_path, logger=None):
    """Download a preprocessed '<tag>_adsorption.json' from the catbench.org mirror.

    Returns a provenance dict on success, None if the mirror does not have the
    dataset (HTTP 404). Raises MirrorUnavailableError on network failure, so the
    caller never falls back to CatHub (and spends API requests) silently.
    """
    import gzip
    import hashlib
    url = f"{MIRROR_URL.rstrip('/')}/{tag}.json.gz"
    try:
        resp = requests.get(url, timeout=120)
    except requests.RequestException as e:
        raise MirrorUnavailableError(
            f"Could not reach the catbench.org mirror ({url}): {type(e).__name__}. "
            f"Check the network, or pass source='cathub' to download from CatHub instead.") from None
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise MirrorUnavailableError(
            f"catbench.org mirror returned HTTP {resp.status_code} for {url}. "
            f"Retry later, or pass source='cathub' to download from CatHub instead.")
    try:
        raw = gzip.decompress(resp.content)
        data = json.loads(raw)
    except Exception as e:
        raise MirrorUnavailableError(f"Mirror file for {tag} is not a valid gzip JSON ({e}).") from None
    n = sum(1 for k in data if not k.startswith("_"))
    if n == 0 or not all("raw" in v for k, v in data.items() if not k.startswith("_")):
        raise MirrorUnavailableError(f"Mirror file for {tag} is not a CatBench adsorption dataset.")
    tmp = dest_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, dest_path)
    prov = {"source": "catbench.org mirror", "url": url, "md5": hashlib.md5(raw).hexdigest(),
            "bytes": len(raw), "reactions": n, "downloaded": time.strftime("%Y-%m-%d %H:%M:%S")}
    if logger:
        logger.info(f"Mirror download: {prov}")
    return prov


def reactions_from_dataset(pub_id, page_size=CATHUB_MAX_PAGE, logger=None, request_budget=CATHUB_DEFAULT_BUDGET):
    """
    Download reactions from CatHub dataset.
    
    Args:
        pub_id: Publication ID or dataset tag
        page_size: Number of reactions per page (CatHub caps it at 200)
        logger: Logger instance for progress tracking
        request_budget: Maximum CatHub requests this download may use. The request
            count is computed from totalCount before downloading; if it exceeds the
            budget the download is refused up front (CatHubBudgetError).
        
    Returns:
        List of reaction data
    """
    page_size = max(1, min(int(page_size), CATHUB_MAX_PAGE))
    total_expected = _count_reactions(pub_id)
    # Measured 2026-09-23: a page that includes structures (InputFile) comes back
    # with ONE reaction no matter what `first` asks for (without structures a page
    # holds up to 200). Plan for the worst case: one request per reaction.
    needed = 1 + total_expected
    if needed > request_budget:
        raise CatHubBudgetError(
            f"Downloading {pub_id} from CatHub needs up to {needed} requests: CatHub returns "
            f"structures one reaction per request ({total_expected} reactions). That is above "
            f"the budget of {request_budget}; CatHub suspends accounts above 500 requests/day. "
            f"Use the catbench.org mirror (source='auto'), or raise request_budget knowingly "
            f"(at 10 requests/minute this takes about {needed // 10 + 1} minutes).")
    if logger:
        logger.info(f"CatHub: {total_expected} reactions, ~{needed} requests planned "
                    f"(page_size={page_size}, budget={request_budget})")
    used = 1
    prev_cursor = None
    reactions = []
    seen_ids = set()
    has_next_page = True
    start_cursor = ""
    page = 0
    total_count = None
    while has_next_page:
        # `order: "id"` enforces a stable total ordering for cursor pagination.
        # Without it, CatHub's default row order is not stable across pages, so
        # paging with `after: endCursor` silently returns some reactions twice and
        # skips an equal number -- yielding non-deterministic, incomplete downloads.
        if used >= request_budget:
            raise CatHubBudgetError(f"Request budget ({request_budget}) exhausted while "
                                    f"downloading {pub_id}; nothing was saved.")
        query = (
            f"""{{
      reactions(pubId: "{pub_id}", first: {page_size}, after: "{start_cursor}", order: "id") {{
        totalCount
        pageInfo {{
          hasNextPage
          hasPreviousPage
          startCursor
          endCursor
        }}
        edges {{
          node {{
            id
            Equation
            reactants
            products
            reactionEnergy
            reactionSystems {{
              name
              systems {{
                energy
                constraints
                InputFile(format: "json")
              }}
            }}
          }}
        }}
      }}
    }}"""
        )
        try:
            data = fetch(query)
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            # One retry for transient server/network errors (never for auth/budget).
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status < 500:
                raise
            if logger:
                logger.warning(f"CatHub request failed ({type(e).__name__}); retrying once.")
            used += 1
            data = fetch(query)
        used += 1
        has_next_page = data["reactions"]["pageInfo"]["hasNextPage"]
        start_cursor = data["reactions"]["pageInfo"]["endCursor"]
        page += 1

        total_count = data["reactions"]["totalCount"]

        # Dedup by CatHub reaction id as a safety net against any residual
        # pagination overlap. With `order: "id"` this should never trigger.
        n_before = len(reactions)
        for edge in data["reactions"]["edges"]:
            node = edge["node"]
            rid = node.get("id")
            if rid is not None and rid in seen_ids:
                continue
            if rid is not None:
                seen_ids.add(rid)
            reactions.append(node)

        # 1.1.5: never trust hasNextPage alone. With first=200 CatHub keeps
        # returning hasNextPage=true and the last page again after everything has
        # been delivered, which would loop until the request budget (or the
        # 500/day account limit) is gone. Stop on any sign of completion.
        new_cursor = data["reactions"]["pageInfo"]["endCursor"]
        if (len(reactions) >= total_expected
                or len(reactions) == n_before
                or new_cursor == prev_cursor):
            has_next_page = False
        prev_cursor = new_cursor

        if logger:
            pct = (len(reactions) / total_count * 100) if total_count else 0.0
            logger.info(f"Downloaded {len(reactions)}/{total_count} unique reactions for {pub_id} "
                        f"({pct:.1f}%, request {used})")

    # Completeness check: the unique download count must match the server-reported
    # total. A mismatch means the dataset is incomplete (do not trust a cached file
    # that fails this check -- delete and re-download).
    if total_count is not None and len(reactions) != total_count:
        msg = (f"Downloaded {len(reactions)} unique reactions but CatHub reports "
               f"totalCount={total_count} for {pub_id}; dataset may be incomplete.")
        if logger:
            logger.warning(msg)
        else:
            print(f"WARNING: {msg}")

    return reactions


def aseify_reactions(reactions):
    """
    Convert reaction data to ASE atoms objects.
    
    Args:
        reactions: List of reaction data from CatHub
    """
    for i, reaction in enumerate(reactions):
        for j, _ in enumerate(reactions[i]["reactionSystems"]):
            system_info = reactions[i]["reactionSystems"][j].pop("systems")

            with io.StringIO() as tmp_file:
                tmp_file.write(system_info.pop("InputFile"))
                tmp_file.seek(0)
                atoms = read(tmp_file, format="json")
                atoms.pbc = True

                # Attach the real FixAtoms constraints from CatHub (if any).
                # gas/bulk legitimately have constraints=None and get no constraint.
                constraints_json = system_info.get("constraints")
                if constraints_json:
                    try:
                        cons_list = json.loads(constraints_json)
                    except (TypeError, ValueError):
                        cons_list = []
                    fix_constraints = []
                    for c in cons_list:
                        if isinstance(c, dict) and c.get("name") == "FixAtoms":
                            idx = sorted(set(int(x) for x in c.get("kwargs", {}).get("indices", [])))
                            if idx:
                                fix_constraints.append(FixAtoms(indices=idx))
                    if fix_constraints:
                        atoms.set_constraint(fix_constraints)

                reactions[i]["reactionSystems"][j]["atoms"] = atoms

            reactions[i]["reactionSystems"][j]["energy"] = system_info["energy"]

        reactions[i]["reactionSystems"] = {
            x["name"]: {"atoms": x["atoms"], "energy": x["energy"]}
            for x in reactions[i]["reactionSystems"]
        }


def _fixed_indices_from_relaxation(slab_atoms, adslab_atoms, adsorbate_indices,
                                   tol=1e-4):
    """Infer which atoms were held fixed during DFT, from geometry alone.

    When a slab/adslab is relaxed with part of the slab frozen (FixAtoms), the
    frozen atoms keep the *exact* same coordinates in the clean slab and in the
    adslab (the fixed coordinates are reused). A free atom relaxes differently
    once an adsorbate is present, so it moves. We therefore compare each adslab
    substrate atom to its nearest same-element clean-slab atom: a match within
    ``tol`` (Angstrom) means that atom never moved -> it was fixed.

    This recovers the deposited FixAtoms set exactly (validated on CatHub
    datasets that *do* carry constraints: per-atom IoU = 1.000 on Comer,
    BackPrediction2018, BajdichWO32018, ...). It is used to reconstruct the
    constraint for datasets where CatHub did not deposit it.

    Args:
        slab_atoms (ase.Atoms): clean slab ("star").
        adslab_atoms (ase.Atoms): adsorbate+slab ("*star").
        adsorbate_indices (list[int]): indices of adsorbate atoms in the adslab.
        tol (float): max displacement (A) to count an atom as fixed. Genuinely
            fixed atoms reuse coordinates verbatim so their displacement is
            exactly 0; free atoms move >~5e-4 A even when far from the adsorbate.
            Default 1e-4 sits in that gap (validated by a tol sweep: deposited-fix
            datasets stay at IoU 1.0 down to 1e-6, while free-but-static atoms only
            drop out below ~1e-3). Going tighter than ~1e-5 starts discarding real
            fixed atoms on datasets whose clean slab carries float-precision noise.

    Returns:
        (slab_fixed, adslab_fixed): sorted index lists. Empty lists mean the
        slab was genuinely unconstrained (relaxed freely). Returns (None, None)
        when detection is not reliable (substrate atom count != clean slab),
        signalling the caller to fall back.
    """
    import numpy as np
    from ase.geometry import find_mic

    ads_set = set(int(i) for i in adsorbate_indices)
    sub_idx = [i for i in range(len(adslab_atoms)) if i not in ads_set]
    # The adslab substrate must correspond 1:1 to the clean slab; if not, the
    # adsorbate-index detection is off and geometric matching is unreliable.
    if len(sub_idx) != len(slab_atoms):
        return None, None

    cell = adslab_atoms.cell
    slab_pos = slab_atoms.positions
    slab_sym = np.array(slab_atoms.get_chemical_symbols())
    ad_sym = np.array(adslab_atoms.get_chemical_symbols())
    ad_pos = adslab_atoms.positions

    adslab_fixed, slab_fixed = [], []
    for ai in sub_idx:
        cand = np.where(slab_sym == ad_sym[ai])[0]
        if len(cand) == 0:
            continue
        disp = slab_pos[cand] - ad_pos[ai]
        mic, _ = find_mic(disp, cell)
        d = np.linalg.norm(mic, axis=1)
        jmin = int(d.argmin())
        if d[jmin] < tol:
            adslab_fixed.append(ai)
            slab_fixed.append(int(cand[jmin]))
    return sorted(set(slab_fixed)), sorted(adslab_fixed)


def cathub_preprocessing(benchmark, adsorbate_integration=None, require_constraints=True,
                         infer_fix_when_missing=True, fix_detect_tol=1e-4,
                         source="auto", api_key=None, request_budget=CATHUB_DEFAULT_BUDGET):
    """
    Download and preprocess CatHub data for MLIP benchmarking.
    
    This function downloads catalysis reaction data from the CatHub database,
    processes the structures and energies, and converts them into a standardized
    format suitable for MLIP benchmarking calculations.
    
    Input Requirements:
        - Internet connection for downloading from CatHub API
        - Valid CatHub publication IDs or dataset tags
        
    Output Files:
        - JSON files: {benchmark}.json containing raw downloaded data
        - JSON files: {benchmark}_adsorption.json with processed benchmark data  
        - YAML files: {output_name}.yml with metadata (for multiple benchmarks)
        - LOG files: cathub_preprocessing.log with processing details
        
    Data Processing Steps:
        1. Download reaction data from CatHub GraphQL API
        2. Convert InputFile JSON strings to ASE Atoms objects
        3. Validate reaction stoichiometry and energy consistency
        4. Filter out incomplete or invalid reactions
        5. Apply adsorbate name integration if specified
        6. Save processed data in standardized JSON format
    
    Args:
        benchmark (str or list): Single benchmark tag or list of benchmark tags.
                               Examples: "AraComputational2022", 
                                       ["AraComputational2022", "AlonsoStrain2023"]
        adsorbate_integration (dict, optional): Mapping for adsorbate name unification.
                                              Format: {"source_name": "target_name"}
                                              Example: {"OH2": "H2O", "H2O2": "OOH"}
        source (str): Where to get the data (new in 1.1.5).
            "auto"   (default) catbench.org mirror first; CatHub only if the mirror
                     does not have the dataset (needs an API key).
            "mirror" mirror only; error if the dataset is not there.
            "cathub" always download from CatHub (needs an API key).
            The mirror holds datasets preprocessed with the default options, so a
            call with non-default preprocessing options goes to CatHub.
        api_key (str, optional): CatHub API key. Falls back to the CATHUB_API_KEY
            environment variable. Sent only as a request header, never logged.
        request_budget (int): Maximum CatHub requests per dataset download
            (default 300; CatHub suspends accounts above 500 requests/day).
                                              
    Raises:
        ValueError: If reaction energy validation fails
        KeyError: If required reaction systems are missing
        ConnectionError: If CatHub API is not accessible
        
    Note:
        The function automatically handles duplicate reaction names and validates
        reaction stoichiometry. Invalid reactions are filtered out with error messages.
    """
    global _api_key
    if source not in ("auto", "mirror", "cathub"):
        raise ValueError(f"source must be 'auto', 'mirror' or 'cathub', got {source!r}")
    save_directory = get_raw_data_directory()
    os.makedirs(save_directory, exist_ok=True)
    
    # Convert single string to list for uniform processing
    benchmarks = [benchmark] if isinstance(benchmark, str) else benchmark

    # --- 1.1.5: catbench.org mirror first -----------------------------------------
    default_options = (adsorbate_integration is None and require_constraints is True
                       and infer_fix_when_missing is True and fix_detect_tol == 1e-4)
    if isinstance(benchmark, str) and source in ("auto", "mirror"):
        out_path = get_raw_data_path(benchmark)
        raw_cache = os.path.join(save_directory, f"{benchmark}.json")
        if os.path.exists(out_path):
            print(f"Processed data already exists at {out_path}")
            return
        if not os.path.exists(raw_cache):
            if not default_options:
                if source == "mirror":
                    raise ValueError("The mirror only holds datasets preprocessed with the default "
                                     "options; use source='cathub' for custom options.")
                print("Non-default preprocessing options: skipping the mirror, using CatHub.")
            else:
                mlog = logging.getLogger(f"catbench_{benchmark}_mirror")
                mlog.setLevel(logging.INFO)
                mlog.handlers.clear()
                mh = logging.FileHandler(os.path.join(save_directory, f"{benchmark}_preprocessing.log"),
                                         mode="w", encoding="utf-8")
                mh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                mlog.addHandler(mh)
                mlog.propagate = False
                try:
                    prov = download_from_mirror(benchmark, out_path, logger=mlog)
                finally:
                    mlog.handlers.clear()
                if prov is not None:
                    print(f"Downloaded {benchmark} from the catbench.org mirror "
                          f"({prov['reactions']} reactions, md5 {prov['md5']}) -> {out_path}")
                    return
                if source == "mirror":
                    raise FileNotFoundError(f"'{benchmark}' is not in the catbench.org mirror.")
                print(f"'{benchmark}' is not in the catbench.org mirror; downloading from CatHub.")
    elif not isinstance(benchmark, str) and source == "mirror":
        raise ValueError("source='mirror' supports a single dataset tag, not a list.")

    # --- CatHub path: needs an API key if anything must be downloaded ---------------
    needs_download = any(not os.path.exists(os.path.join(save_directory, f"{b}.json"))
                         for b in benchmarks)
    if needs_download and not os.path.exists(get_raw_data_path(
            benchmark if isinstance(benchmark, str) else "multiple_tag")):
        key = _resolve_api_key(api_key)
        if key is None:
            first_missing = next(b for b in benchmarks
                                 if not os.path.exists(os.path.join(save_directory, f"{b}.json")))
            raise CatHubAuthError(_missing_key_message(first_missing))
        _api_key = key
    try:
        return _cathub_preprocessing_impl(benchmark, benchmarks, save_directory, adsorbate_integration,
                                          require_constraints, infer_fix_when_missing, fix_detect_tol,
                                          request_budget)
    finally:
        _api_key = None


def _cathub_preprocessing_impl(benchmark, benchmarks, save_directory, adsorbate_integration,
                               require_constraints, infer_fix_when_missing, fix_detect_tol,
                               request_budget):
    """Download (CatHub) + preprocess. Split out of cathub_preprocessing in 1.1.5."""

    # Check if any downloads are needed and setup logging if so
    download_needed = any(not os.path.exists(os.path.join(save_directory, f"{bench}.json")) 
                         for bench in benchmarks)
    
    logger = None
    if download_needed:
        log_name = benchmark if isinstance(benchmark, str) else "multiple_tag"
        log_file = os.path.join(save_directory, f"{log_name}_preprocessing.log")
        logger = logging.getLogger(f"catbench_{log_name}")
        logger.setLevel(logging.INFO)
        # Remove existing handlers to prevent duplication
        if logger.hasHandlers():
            logger.handlers.clear()
        handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        logger.info(f"Starting CatHub data download for benchmark: {benchmark}")
    
    # Initialize combined data structure
    combined_reactions = []
    
    for bench in benchmarks:
        path_json = os.path.join(save_directory, f"{bench}.json")
        # Create separate logger/handler for each benchmark
        bench_logger = None
        log_file = os.path.join(save_directory, f"{bench}_preprocessing.log")
        bench_logger = logging.getLogger(f"catbench_{bench}")
        bench_logger.setLevel(logging.INFO)
        if bench_logger.hasHandlers():
            bench_logger.handlers.clear()
        handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        bench_logger.addHandler(handler)
        bench_logger.propagate = False
        bench_logger.info(f"Starting CatHub data download for benchmark: {bench}")
        
        # Get reactions for benchmark (preliminary download)
        if not os.path.exists(path_json):
            bench_logger.info(f"Downloading reactions for benchmark: {bench}")
            raw_reactions = reactions_from_dataset(bench, logger=bench_logger,
                                                   request_budget=request_budget)
            bench_logger.info(f"Download completed for {bench}: {len(raw_reactions)} reactions")
            raw_reactions_json = {"raw_reactions": raw_reactions}
            save_json(raw_reactions_json, path_json, use_numpy_encoder=False)
            bench_logger.info(f"Saved raw data to {path_json}")
        else:
            with open(path_json, "r") as file:
                raw_reactions_json = json.load(file)
        combined_reactions.extend(raw_reactions_json["raw_reactions"])
                    # Remove handlers for each benchmark to prevent memory leaks
        bench_logger.handlers.clear()
    
    # Generate output filename based on input type
    if isinstance(benchmark, str):
        output_name = benchmark
    else:
        output_name = "multiple_tag"
        # Save benchmark information to yaml file
        benchmark_info = {
            "benchmarks": sorted(benchmarks),
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_reactions": len(combined_reactions)
        }
        yaml_path = os.path.join(save_directory, f"{output_name}.yml")
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(benchmark_info, yaml_file, default_flow_style=False)
    
    # JSON format with _catbench suffix
    path_output = get_raw_data_path(output_name)
    
    if not os.path.exists(path_output):
        # Setup logging if not already done (e.g., no download was needed)
        if logger is None:
            log_file = os.path.join(save_directory, f"{output_name}_preprocessing.log")
            # Use a dedicated named logger with an explicit FileHandler instead of
            # logging.basicConfig, which mutates the root logger (no-op if already
            # configured, double-logs in notebooks).
            logger = logging.getLogger(f"catbench_{output_name}")
            logger.setLevel(logging.INFO)
            if logger.hasHandlers():
                logger.handlers.clear()
            handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            logger.propagate = False
            logger.info(f"Starting CatHub preprocessing for benchmark: {benchmark}")
        
        # Continue with processing logs
        logger.info(f"Starting data processing for benchmark: {benchmark}")
        if adsorbate_integration:
            logger.info(f"Adsorbate integration mapping: {adsorbate_integration}")
        
        if isinstance(benchmark, str):
            logger.info(f"Processing single benchmark: {benchmark}")
        else:
            logger.info(f"Processing multiple benchmarks: {benchmarks}")
            logger.info(f"Benchmark metadata saved to {os.path.join(save_directory, f'{output_name}.yml')}")
        
        logger.info(f"Total combined reactions to process: {len(combined_reactions)}")
        logger.info(f"Data processing output will be saved to {path_output}")
        
        # Process combined reactions
        dat = copy.deepcopy(combined_reactions)

        # Deterministic dedup safety net. New downloads carry a CatHub `id`; older
        # cached raw files (downloaded before the ordering fix) do not, so fall back
        # to a content signature. This removes the duplicate inflation a previous
        # unstable-pagination download may have baked into a cached `{bench}.json`.
        # NOTE: dedup cannot recover reactions that were *dropped* -- if a cache is
        # incomplete, delete it and re-download.
        def _reaction_signature(rxn):
            rid = rxn.get("id")
            if rid is not None:
                return ("id", rid)
            systems = tuple(sorted(
                (rs.get("name"), round(rs.get("systems", {}).get("energy", 0.0), 4))
                for rs in rxn.get("reactionSystems", [])
            ))
            return ("content", rxn.get("Equation"), round(rxn.get("reactionEnergy", 0.0), 4), systems)

        deduped = []
        seen_sig = set()
        dup_dropped = 0
        for rxn in dat:
            sig = _reaction_signature(rxn)
            if sig in seen_sig:
                dup_dropped += 1
                continue
            seen_sig.add(sig)
            deduped.append(rxn)
        if dup_dropped:
            logger.warning(f"Dropped {dup_dropped} duplicate reaction(s) from input "
                           f"({len(dat)} -> {len(deduped)}). Source download was likely "
                           f"produced by a pre-fix unstable pagination; consider re-downloading "
                           f"to also recover any reactions that were skipped.")
        dat = deduped

        logger.info("Converting reaction data to ASE atoms objects...")
        aseify_reactions(dat)

        data_total = {}
        tags = []
        
        logger.info(f"Processing {len(dat)} reactions for energy validation...")

        for i, _ in enumerate(dat):
            try:
                input = {}
                reactants_json = dat[i]["reactants"]
                reactants_dict = json.loads(reactants_json)

                products_json = dat[i]["products"]
                products_dict = json.loads(products_json)



                # Generate tag safely - use 'star' if available, otherwise use first available system
                if "star" in dat[i]["reactionSystems"]:
                    sym = dat[i]["reactionSystems"]["star"]["atoms"].get_chemical_formula()
                else:
                    # Use first available system for tag generation
                    first_key = next(iter(dat[i]["reactionSystems"]))
                    sym = dat[i]["reactionSystems"][first_key]["atoms"].get_chemical_formula()
                
                reaction_name = dat[i]["Equation"]
                # Track the *base* tag in `tags` so the duplicate count stays
                # consistent. The displayed tag gets a deterministic `_N` suffix
                # only for genuine collisions (distinct reactions sharing the same
                # formula + equation). `base_tag` is what we add/remove for counting.
                base_tag = sym + "_" + reaction_name
                count = tags.count(base_tag)
                tags.append(base_tag)
                tag = base_tag if count == 0 else f"{base_tag}_{count}"



                # Step 1: Add all structures from reactants and products dictionaries as-is
                for key in dat[i]["reactionSystems"]:
                    if key in reactants_dict:
                        input[key] = {
                            "stoi": -reactants_dict[key],
                            "atoms": dat[i]["reactionSystems"][key]["atoms"],
                            "energy_ref": dat[i]["reactionSystems"][key]["energy"],
                        }
                    elif key in products_dict:
                        input[key] = {
                            "stoi": products_dict[key],
                            "atoms": dat[i]["reactionSystems"][key]["atoms"],
                            "energy_ref": dat[i]["reactionSystems"][key]["energy"],
                        }

                # Filter for single adsorbate reactions only
                star_structures = [key for key in input.keys() if "star" in key]
                if len(star_structures) != 2:
                    logger.info(f"Filtered - {tag}: Multi-adsorbate reaction (found {len(star_structures) - 1} adslab structures, expected 1)")
                    continue

                # Step 2: First energy validation with original coefficients
                energy_check = 0
                for structure in input:
                    energy_check += (
                        input[structure]["energy_ref"] * input[structure]["stoi"]
                    )
                
                validation_passed = False
                if abs(dat[i]["reactionEnergy"] - energy_check) <= 0.001:
                    # Validation passed with original coefficients
                    validation_passed = True
                    logger.debug(f"Energy validation passed for {tag} with original coefficients")
                else:
                    # Step 3: Try with adslab coefficient = 1
                    logger.debug(f"First validation failed for {tag}, trying with adslab coefficient = 1")
                    
                    # Find the adslab structure (not "star")
                    adslab_key = None
                    for key in input.keys():
                        if "star" in key and key != "star":
                            adslab_key = key
                            break
                    
                    if adslab_key:
                        original_coeff = input[adslab_key]["stoi"]
                        input[adslab_key]["stoi"] = 1  # Try with coefficient 1
                        
                        # Recalculate energy with corrected coefficient
                        energy_check_retry = 0
                        for structure in input:
                            energy_check_retry += (
                                input[structure]["energy_ref"] * input[structure]["stoi"]
                            )
                        
                        if abs(dat[i]["reactionEnergy"] - energy_check_retry) <= 0.001:
                            # Validation passed with coefficient = 1
                            validation_passed = True
                            logger.info(f"Energy validation passed for {tag} with adslab coefficient changed from {original_coeff} to 1")
                        else:
                            # Revert to original coefficient if still fails
                            input[adslab_key]["stoi"] = original_coeff
                
                # Step 4: Save or filter based on validation result
                if validation_passed:
                    data_total[tag] = {}
                    data_total[tag]["raw"] = input
                    data_total[tag]["ref_ads_eng"] = dat[i]["reactionEnergy"]
                else:
                    # Both validations failed
                    logger.info(f"Filtered - {tag}: Stoichiometry inconsistency even after trying coefficient adjustment")
                    logger.debug(f"  CatHub energy: {dat[i]['reactionEnergy']:.6f} eV")
                    logger.debug(f"  Original calculation: {energy_check:.6f} eV (diff: {abs(dat[i]['reactionEnergy'] - energy_check):.6f} eV)")
                    if adslab_key:
                        logger.debug(f"  With adslab coeff=1: {energy_check_retry:.6f} eV (diff: {abs(dat[i]['reactionEnergy'] - energy_check_retry):.6f} eV)")
                    
                    # Roll back the count bookkeeping using the base tag. The old
                    # code removed the suffixed `tag` (e.g. "X_1"), which was never
                    # added to `tags`, so removal silently failed and corrupted the
                    # duplicate count for later collisions.
                    if base_tag in tags:
                        tags.remove(base_tag)
                    continue

                # Apply adsorbate integration if specified
                if adsorbate_integration:
                    integration_applied = False
                    for key in list(data_total[tag]["raw"].keys()):
                        if "star" in key and key != "star":
                            adsorbate = key[:-4]
                            if adsorbate in adsorbate_integration:
                                integrated_key = f"{adsorbate_integration[adsorbate]}star"
                                data_total[tag]["raw"][integrated_key] = data_total[tag]["raw"].pop(key)
                                logger.debug(f"Integrated adsorbate in {tag}: {key} → {integrated_key}")
                                integration_applied = True
                    if integration_applied:
                        logger.info(f"Applied adsorbate integration to reaction: {tag}")
                
                # Add adsorbate indices detection using common utility
                from catbench.utils.data_utils import detect_adsorbate_indices
                
                # Find slab and adslab structures
                slab_key = "star"
                adslab_key = None
                for key in data_total[tag]["raw"].keys():
                    if "star" in key and key != "star":
                        adslab_key = key
                        break
                
                if slab_key in data_total[tag]["raw"] and adslab_key:
                    slab_atoms = data_total[tag]["raw"][slab_key]["atoms"]
                    adslab_atoms = data_total[tag]["raw"][adslab_key]["atoms"]
                    
                    # Use common detection function
                    adsorbate_indices = detect_adsorbate_indices(slab_atoms, adslab_atoms)
                    
                    # Store adsorbate indices
                    data_total[tag]["adsorbate_indices"] = adsorbate_indices
                    logger.debug(f"Detected adsorbate indices for {tag}: {adsorbate_indices}")
                else:
                    # If we can't detect, set empty list
                    data_total[tag]["adsorbate_indices"] = []
                    logger.warning(f"Could not detect adsorbate indices for {tag}")

                # Constraint handling. CatHub normally deposits FixAtoms for
                # slabs/adslabs, but a sizeable fraction of datasets do not.
                # Policy:
                #   - if the star/*star already carries FixAtoms -> keep it as-is.
                #   - if it is missing and infer_fix_when_missing -> reconstruct
                #     the fixed-atom set from geometry (clean slab vs adslab; the
                #     frozen atoms never move). Inject it so the structure becomes
                #     self-describing and runs identically to a deposited one.
                #     An empty inferred set means the slab was genuinely free.
                #   - if inference is unreliable (atom-count mismatch) -> fall
                #     back to require_constraints (raise & skip) or keep flagged.
                # gas/bulk references legitimately carry no constraints (exempt).
                def _has_fixatoms(atoms):
                    return any(
                        isinstance(c, FixAtoms) and len(c.get_indices()) > 0
                        for c in atoms.constraints
                    )

                raw_structs = data_total[tag]["raw"]
                missing = [k for k in raw_structs
                           if "star" in k and not _has_fixatoms(raw_structs[k]["atoms"])]

                if not missing:
                    data_total[tag]["constraint_source"] = "deposited"
                else:
                    injected = False
                    if (infer_fix_when_missing and slab_key in raw_structs
                            and adslab_key in raw_structs):
                        slab_fx, adslab_fx = _fixed_indices_from_relaxation(
                            raw_structs[slab_key]["atoms"],
                            raw_structs[adslab_key]["atoms"],
                            data_total[tag].get("adsorbate_indices", []),
                            tol=fix_detect_tol,
                        )
                        if slab_fx is not None:  # detection ran (possibly empty=free)
                            if adslab_key in missing and adslab_fx:
                                raw_structs[adslab_key]["atoms"].set_constraint(
                                    FixAtoms(indices=adslab_fx))
                            if slab_key in missing and slab_fx:
                                raw_structs[slab_key]["atoms"].set_constraint(
                                    FixAtoms(indices=slab_fx))
                            data_total[tag]["constraint_source"] = (
                                "geometry_inferred" if adslab_fx else "free")
                            logger.debug(
                                f"{tag}: inferred fix (slab={len(slab_fx)}, "
                                f"adslab={len(adslab_fx)} atoms)")
                            injected = True
                    if not injected:
                        if require_constraints:
                            data_total.pop(tag, None)
                            raise ValueError(
                                f"CatHub data gap: reaction '{tag}' has a star/*star "
                                f"with no FixAtoms and the fixed set could not be "
                                f"inferred (substrate/clean-slab atom-count mismatch). "
                                f"Cannot benchmark reliably. (gas/bulk are exempt.)"
                            )
                        data_total[tag]["constraint_source"] = "undetermined_kept"
                        logger.warning(
                            f"{tag}: missing FixAtoms and inference unavailable; "
                            f"kept without constraint (will relax fully unconstrained).")

                # Log successful processing
                logger.debug(f"Successfully processed reaction: {tag}")
                logger.debug(f"  Reaction energy: {dat[i]['reactionEnergy']:.6f} eV")
                logger.debug(f"  Number of structures: {len(data_total[tag]['raw'])}")

            except Exception as e:
                logger.error(f"Unexpected error processing reaction {i+1}/{len(dat)}: {e}")
                logger.error(f"Reaction tag: {tag if 'tag' in locals() else 'Unknown'}")
                logger.error(f"Traceback: {traceback.format_exc()}")

        # Processing complete - show detailed statistics
        filtered_count = len(dat) - len(data_total)
        success_rate = (len(data_total) / len(dat) * 100) if len(dat) > 0 else 0
        
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total reactions attempted: {len(dat)}")
        logger.info(f"Successfully processed: {len(data_total)} ({success_rate:.1f}%)")
        logger.info(f"Filtered out: {filtered_count} ({filtered_count/len(dat)*100:.1f}%)")
        logger.info("=" * 60)
        
        # Also print to console for immediate visibility
        print("\n" + "=" * 60)
        print("CATHUB DATA PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Successfully processed: {len(data_total)}/{len(dat)} reactions ({success_rate:.1f}%)")
        print(f"Filtered out: {filtered_count} reactions")
        print(f"Output file: {path_output}")
        print("=" * 60 + "\n")
        
        logger.info(f"Saving processed data to {path_output}")
        
        # JSON format
        from catbench.utils.data_utils import save_catbench_json
        save_catbench_json(data_total, path_output)
        logger.info("Data processing and saving completed successfully!")
    else:
        print(f"Processed data already exists at {path_output}")
        print("   Skipping processing to avoid overwriting existing data.")
        print("   Delete the file if you want to reprocess the data.")


def download(benchmark_tags, api_key=None, request_budget=CATHUB_DEFAULT_BUDGET):
    """
    Download raw reaction data from CatHub (without processing).
    
    Args:
        benchmark_tags: Single tag or list of tags
        api_key: CatHub API key (falls back to CATHUB_API_KEY)
        request_budget: Maximum CatHub requests per tag
        
    Returns:
        List of raw reaction data
    """
    global _api_key
    if isinstance(benchmark_tags, str):
        benchmark_tags = [benchmark_tags]
    key = _resolve_api_key(api_key)
    if key is None:
        raise CatHubAuthError(_missing_key_message(benchmark_tags[0]))
    _api_key = key
    try:
        all_reactions = []
        for tag in benchmark_tags:
            reactions = reactions_from_dataset(tag, request_budget=request_budget)
            all_reactions.extend(reactions)
    finally:
        _api_key = None
    return all_reactions 