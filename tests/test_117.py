"""catbench 1.1.7: bulk CatHub download (structures per publication), no unstamped
resume for multi-head models."""
import json

import pytest
from ase.calculators.emt import EMT

from catbench.adsorption.data import cathub as ch
from tests.test_115 import _counting
from tests.test_116 import FakeMACE, _old_result


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CATBENCH_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(ch, "CATHUB_MIN_INTERVAL", 0)


def _rxn(i, uids):
    return {"node": {"id": f"r{i}", "Equation": "O(g) + * -> O*", "reactants": "{}", "products": "{}",
                     "reactionEnergy": -1.0,
                     "reactionSystems": [{"name": n, "systems": {"uniqueId": u}} for n, u in uids]}}


def _system(u):
    return {"uniqueId": u, "energy": -1.5, "constraints": None, "InputFile": '{"1": {}, "ids": [1], "nextid": 2}'}


def _fake_cathub(n, systems, calls):
    rx = [_rxn(i, [("star", "slab"), ("Ostar", f"ads{i}"), ("Ogas", "gas")]) for i in range(n)]

    def fake(query):
        calls.append(query)
        assert "InputFile" not in query or "publications(" in query, "reaction pages must not ask for structures"
        if "first: 1)" in query:
            return {"reactions": {"totalCount": n}}
        if "publications(" in query:
            return {"publications": {"edges": [{"node": {"pubId": "P2026", "systems": systems}}]}}
        start = 0 if 'after: ""' in query else 200
        page = rx[start:start + 200]
        return {"reactions": {"totalCount": n, "edges": page,
                              "pageInfo": {"hasNextPage": start + 200 < n, "endCursor": f"c{start}"}}}
    return fake


def test_bulk_download_joins_structures_by_unique_id(monkeypatch):
    calls = []
    systems = [_system("slab"), _system("gas")] + [_system(f"ads{i}") for i in range(250)]
    monkeypatch.setattr(ch, "fetch", _counting(_fake_cathub(250, systems, calls)))
    got = ch.reactions_from_dataset("P2026", request_budget=10)
    assert len(got) == 250
    assert len(calls) == 4                                   # probe + 2 pages + 1 publication
    rs = got[7]["reactionSystems"]
    assert [r["name"] for r in rs] == ["star", "Ostar", "Ogas"]
    assert set(rs[1]["systems"]) == {"energy", "constraints", "InputFile"}   # same shape as 1.1.6


def test_bulk_download_refuses_missing_structures(monkeypatch):
    calls = []
    systems = [_system("slab"), _system("gas")] + [_system(f"ads{i}") for i in range(9)]   # ads9 missing
    monkeypatch.setattr(ch, "fetch", _counting(_fake_cathub(10, systems, calls)))
    with pytest.raises(RuntimeError, match="not in the publication"):
        ch.reactions_from_dataset("P2026", request_budget=10)


def test_multi_head_model_refuses_unstamped_resume(tmp_path):
    from catbench.adsorption import AdsorptionCalculation
    d = tmp_path / "result" / "M"
    _old_result(d, {"dispersion": {"method": "none"}})          # pre-1.1.6: no calculator stamp
    run = AdsorptionCalculation([FakeMACE("oc20_usemppbe")], mlip_name="M", benchmark="B")
    with pytest.raises(RuntimeError, match="start fresh"):
        run._load_existing_results(str(d))
    ok = AdsorptionCalculation([FakeMACE("oc20_usemppbe")], mlip_name="M", benchmark="B",
                               allow_unstamped_resume=True)
    final, *_ = ok._load_existing_results(str(d))
    assert "r1" in final


def test_single_head_model_still_resumes_unstamped(tmp_path):
    from catbench.adsorption import AdsorptionCalculation
    d = tmp_path / "result" / "M"
    _old_result(d, {"dispersion": {"method": "none"}})
    run = AdsorptionCalculation([EMT()], mlip_name="M", benchmark="B")
    final, *_ = run._load_existing_results(str(d))
    assert "r1" in final
    assert run._run_metadata()["calculator"]["unverified_prior_reactions"] == 1


def test_multi_head_drops_unstamped_structure_cache(tmp_path):
    from catbench.adsorption import AdsorptionCalculation
    d = tmp_path / "result" / "M"
    d.mkdir(parents=True)
    run = AdsorptionCalculation([FakeMACE("oc20_usemppbe")], mlip_name="M", benchmark="B")
    (d / "M_structure_cache.json").write_text(json.dumps({"__relax_config__": run._relax_sig()[:5], "k": 1}))
    *_, cache = run._load_existing_results(str(d))
    assert cache == {}                                       # 1.1.5 cache: head unknown -> recompute


class FakeSingleHeadMACE(EMT):   # mace-mpa-0: head='Default', one head
    def __init__(self):
        super().__init__()
        self.available_heads = ["Default"]
        self.head = "Default"


def test_single_head_mace_still_resumes_unstamped(tmp_path):
    from catbench.adsorption import AdsorptionCalculation
    d = tmp_path / "result" / "M"
    _old_result(d, {"dispersion": {"method": "none"}})
    run = AdsorptionCalculation([FakeSingleHeadMACE()], mlip_name="M", benchmark="B")
    final, *_ = run._load_existing_results(str(d))
    assert "r1" in final
