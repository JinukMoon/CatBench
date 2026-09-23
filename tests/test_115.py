"""1.1.5 regressions: FixAtoms-preserving dedup, CatHub key/mirror/budget handling,
_failures-tolerant analysis, dispersion/device stamps."""
import gzip
import json
import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.calculators.mixing import SumCalculator
from ase.constraints import FixAtoms

import catbench
from catbench.adsorption.data import cathub as ch
from catbench.utils.data_utils import save_catbench_json, load_catbench_json

FAKE_KEY = "FAKEKEY-do-not-leak-1234567890"


def _fixed(atoms):
    return sorted(int(i) for c in atoms.constraints if isinstance(c, FixAtoms) for i in c.get_indices())


# --- G: the 2026-06 data incident -------------------------------------------------
def test_dedup_keeps_per_reaction_fixatoms(tmp_path):
    """Same slab geometry + energy but different FixAtoms in two reactions must NOT be
    merged into one stored copy (that swapped constraints in 445 published reactions)."""
    base = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    fixed_slab = base.copy()
    fixed_slab.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    free_slab = base.copy()
    data = {}
    for name, slab in (("fixed", fixed_slab), ("free", free_slab)):
        ads = slab.copy()
        ads += Atoms("O", positions=[[1.4, 0.8, ads.positions[:, 2].max() + 1.8]])
        data[name] = {"raw": {"star": {"atoms": slab, "energy_ref": -1.0, "stoi": -1},
                              "Ostar": {"atoms": ads, "energy_ref": -2.0, "stoi": 1}},
                      "ref_ads_eng": -1.0, "adsorbate_indices": [len(slab)],
                      "constraint_source": "deposited" if name == "fixed" else "free"}
    path = str(tmp_path / "X_adsorption.json")
    save_catbench_json(data, path)
    back = load_catbench_json(path)
    assert _fixed(back["fixed"]["raw"]["star"]["atoms"]) == [0, 1, 2, 3]
    assert _fixed(back["free"]["raw"]["star"]["atoms"]) == []
    assert _fixed(back["fixed"]["raw"]["Ostar"]["atoms"]) == [0, 1, 2, 3]
    assert _fixed(back["free"]["raw"]["Ostar"]["atoms"]) == []


# --- A/9: CatHub access ------------------------------------------------------------
class _Resp:
    def __init__(self, status, payload=None, content=b""):
        self.status_code = status
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}", response=self)


def test_key_is_sent_as_header_and_never_leaks(monkeypatch):
    seen = {}

    def post(url, json=None, headers=None, timeout=None, allow_redirects=None):
        seen.update(url=url, headers=headers, allow_redirects=allow_redirects)
        return _Resp(401)

    monkeypatch.setattr(ch.requests, "post", post)
    monkeypatch.setattr(ch, "_api_key", FAKE_KEY)
    monkeypatch.setattr(ch, "CATHUB_MIN_INTERVAL", 0)
    with pytest.raises(ch.CatHubAuthError) as e:
        ch.fetch("{ x }")
    assert seen["url"].startswith("https://")
    assert seen["headers"] == {"X-API-Key": FAKE_KEY}
    assert seen["allow_redirects"] is False
    assert FAKE_KEY not in str(e.value) and FAKE_KEY not in repr(e.value)
    assert FAKE_KEY not in seen["url"]


def test_resolve_key_precedence(monkeypatch):
    monkeypatch.setenv("CATHUB_API_KEY", "from-env")
    assert ch._resolve_api_key() == "from-env"
    assert ch._resolve_api_key("explicit") == "explicit"
    with pytest.raises(ch.CatHubAuthError):
        ch._resolve_api_key("   ")
    monkeypatch.delenv("CATHUB_API_KEY")
    assert ch._resolve_api_key() is None


def _mirror_payload():
    slab = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
    slab.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))
    ads = slab.copy()
    ads += Atoms("O", positions=[[1.4, 0.8, ads.positions[:, 2].max() + 1.8]])
    return {"r1": {"raw": {"star": {"atoms": slab, "energy_ref": -1.0, "stoi": -1},
                           "Ostar": {"atoms": ads, "energy_ref": -2.0, "stoi": 1}},
                   "ref_ads_eng": -1.0, "adsorbate_indices": [12], "constraint_source": "deposited"}}


def test_mirror_first_needs_no_key(tmp_path, monkeypatch):
    src = tmp_path / "src.json"
    save_catbench_json(_mirror_payload(), str(src))
    blob = gzip.compress(src.read_bytes())
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CATHUB_API_KEY", raising=False)
    monkeypatch.setattr(ch.requests, "get", lambda url, timeout=None: _Resp(200, content=blob))
    monkeypatch.setattr(ch.requests, "post", lambda *a, **k: pytest.fail("CatHub must not be called"))
    ch.cathub_preprocessing("TESTSET")
    out = tmp_path / "raw_data" / "TESTSET_adsorption.json"
    assert out.exists()
    assert _fixed(load_catbench_json(str(out))["r1"]["raw"]["star"]["atoms"]) == [0, 1, 2, 3]


def test_not_in_mirror_and_no_key_explains(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CATHUB_API_KEY", raising=False)
    monkeypatch.setattr(ch.requests, "get", lambda url, timeout=None: _Resp(404))
    monkeypatch.setattr(ch.requests, "post", lambda *a, **k: pytest.fail("no key -> no request"))
    with pytest.raises(ch.CatHubAuthError) as e:
        ch.cathub_preprocessing("NEWSET2026")
    msg = str(e.value)
    assert "not available from the catbench.org mirror" in msg
    assert ch.CATHUB_KEY_URL in msg and "CATHUB_API_KEY" in msg


def test_mirror_network_error_does_not_fall_back_to_cathub(tmp_path, monkeypatch):
    import requests
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CATHUB_API_KEY", FAKE_KEY)

    def boom(url, timeout=None):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(ch.requests, "get", boom)
    monkeypatch.setattr(ch.requests, "post", lambda *a, **k: pytest.fail("must not spend CatHub requests"))
    with pytest.raises(ch.MirrorUnavailableError):
        ch.cathub_preprocessing("ANYSET")


def test_budget_refused_before_download(monkeypatch):
    calls = []

    def fake_fetch(query):
        calls.append(query)
        return {"reactions": {"totalCount": 88587}}

    monkeypatch.setattr(ch, "fetch", fake_fetch)
    with pytest.raises(ch.CatHubBudgetError) as e:
        ch.reactions_from_dataset("MamunHighT2019", request_budget=300)
    assert len(calls) == 1          # only the totalCount probe was spent
    assert "88588" in str(e.value)  # one request per reaction + 1 probe


def test_pagination_stops_even_if_server_keeps_saying_next_page(monkeypatch):
    """Observed 2026-09-23 with first=200: CatHub kept hasNextPage=true and re-sent the
    last page after all 325 reactions were delivered -> 67 wasted requests."""
    calls = []
    page1 = [{"node": {"id": f"r{i}"}} for i in range(200)]
    page2 = [{"node": {"id": f"r{i}"}} for i in range(200, 325)]

    def fake_fetch(query):
        calls.append(query)
        if "first: 1)" in query:
            return {"reactions": {"totalCount": 325}}
        edges = page1 if len(calls) == 2 else page2
        return {"reactions": {"totalCount": 325, "edges": edges,
                              "pageInfo": {"hasNextPage": True, "endCursor": "c%d" % min(len(calls), 3)}}}

    monkeypatch.setattr(ch, "fetch", fake_fetch)
    got = ch.reactions_from_dataset("ComerGeneralized2024", request_budget=400)
    assert len(got) == 325
    assert len(calls) == 3          # probe + 2 pages, not a loop to the budget


# --- D: analysis must survive _failures and report coverage ------------------------
def _dataset(tmp):
    os.makedirs(os.path.join(tmp, "raw_data"), exist_ok=True)
    data = {}
    for i in range(3):
        slab = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
        slab.set_constraint(FixAtoms(mask=[a.position[2] < slab.positions[:, 2].mean() for a in slab]))
        ads = slab.copy()
        ads += Atoms("O", positions=[[ads.cell[0, 0] / 2, ads.cell[1, 1] / 2,
                                      ads.positions[:, 2].max() + 1.7 + 0.1 * i]])
        data["O_%d" % i] = {"raw": {"star": {"atoms": slab, "energy_ref": -1.0, "stoi": -1},
                                    "Ostar": {"atoms": ads, "energy_ref": 0.0, "stoi": 1}},
                            "ref_ads_eng": 0.0, "adsorbate_indices": [len(slab)],
                            "constraint_source": "deposited"}
    # one reaction whose slab key is missing -> fails at setup in every model
    data["broken"] = {"raw": {"Ostar": {"atoms": data["O_0"]["raw"]["Ostar"]["atoms"],
                                        "energy_ref": 0.0, "stoi": 1}},
                      "ref_ads_eng": 0.0, "adsorbate_indices": [12], "constraint_source": "deposited"}
    save_catbench_json(data, os.path.join(tmp, "raw_data", "COV_adsorption.json"))


def test_failures_do_not_crash_analysis_and_are_reported(tmp_path, monkeypatch, capsys):
    openpyxl = pytest.importorskip("openpyxl")
    from catbench.adsorption import AdsorptionCalculation, AdsorptionAnalysis
    monkeypatch.chdir(tmp_path)
    _dataset(str(tmp_path))
    AdsorptionCalculation([EMT(), EMT(), EMT()], mlip_name="EMT", benchmark="COV",
                          save_files=False, n_crit_relax=60).run()
    res = json.load(open(tmp_path / "result" / "EMT" / "EMT_result.json"))
    assert "broken" in res["_failures"] and res["_failures"]["broken"]["stage"]
    st = res["calculation_settings"]
    assert st["catbench_version"] == catbench.__version__
    assert st["n_reactions_input"] == 4 and len(st["input_md5"]) == 32
    assert st["dispersion"]["method"] == "none"
    assert "name" in st["device"]
    out = capsys.readouterr().out
    assert "FAILED" in out

    AdsorptionAnalysis().analysis()          # 1.1.4 crashed here with KeyError: 'final'
    xlsx = [f for f in os.listdir(tmp_path) if f.endswith("_Benchmarking_Analysis.xlsx")][0]
    wb = openpyxl.load_workbook(tmp_path / xlsx, data_only=True)
    rows = list(wb["Coverage"].iter_rows(values_only=True))
    hdr, row = rows[0], rows[1]
    got = dict(zip(hdr, row))
    assert got["Num_input"] == 4 and got["Num_succeeded"] == 3 and got["Num_failed"] == 1


# --- C: dispersion introspection ----------------------------------------------------
class D3Calculator(EMT):   # stand-in with the attributes of catbench's CUDA D3Calculator
    def __init__(self):
        super().__init__()
        self.damp_name, self.func_name, self.rthr, self.cnthr = "damp_zero", "pbe", 9000, 1600


def test_dispersion_is_introspected_not_guessed():
    from catbench.adsorption import AdsorptionCalculation
    plain = AdsorptionCalculation([EMT()], mlip_name="X_D3", benchmark="B")   # name says D3 ...
    assert plain._dispersion_info()["method"] == "none"                     # ... but none attached
    wrapped = AdsorptionCalculation([SumCalculator([EMT(), D3Calculator()])], mlip_name="X", benchmark="B")
    info = wrapped._dispersion_info()
    assert info["method"] == "D3" and info["damping"] == "damp_zero" and info["functional"] == "pbe"
    assert plain._relax_sig() != wrapped._relax_sig()   # cache must not be shared across D3 on/off


def test_get_benchmark_network_error_does_not_fall_back_to_cathub(tmp_path, monkeypatch):
    import requests
    from catbench.adsorption.data import zenodo as zmod
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(zmod, "_zenodo_latest_files", lambda force=False: {})

    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(zmod.requests, "head", boom)
    monkeypatch.setattr(ch, "cathub_preprocessing", lambda *a, **k: pytest.fail("silent CatHub fallback"))
    with pytest.raises(ch.MirrorUnavailableError):
        zmod.get_benchmark("SomeSet2024")
