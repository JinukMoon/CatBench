"""catbench 1.1.6: CatHub API move, calculator head/task stamp, clean-slab-less reactions."""
import json
import os

import pytest
from ase.calculators.emt import EMT
from ase.calculators.mixing import SumCalculator

from catbench.adsorption.data import cathub as ch
from tests.test_115 import D3Calculator, _dataset


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CATBENCH_STATE_DIR", str(tmp_path / "state"))


class FakeMACE(EMT):     # stand-in exposing mace-torch MACECalculator's head attributes
    def __init__(self, head):
        super().__init__()
        self.available_heads = ["matpes_r2scan", "oc20_usemppbe", "omat_pbe"]
        self.head = head


class FakeUMA(EMT):      # fairchem FAIRChemCalculator exposes the task as task_name
    def __init__(self, task):
        super().__init__()
        self.task_name = task


def test_cathub_api_moved_to_slac():
    assert ch.GRAPHQL == "https://api-catalysis-hub.slac.stanford.edu/graphql"
    assert ch.CATHUB_KEY_URL == "https://api-catalysis-hub.slac.stanford.edu/auth/login"
    assert "api.catalysis-hub.org" not in ch.GRAPHQL


def test_no_key_message_blames_the_mirror_only_when_it_was_asked(tmp_path, monkeypatch):
    from catbench.adsorption import cathub_preprocessing
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CATHUB_API_KEY", raising=False)
    with pytest.raises(ch.CatHubAuthError) as e:
        cathub_preprocessing("SomeDataset2026", source="cathub")       # mirror never consulted
    msg = str(e.value)
    assert "mirror" not in msg and "requires an API key" in msg and ch.CATHUB_KEY_URL in msg


def test_calculator_head_is_read_from_the_calculator_not_the_name():
    from catbench.adsorption import AdsorptionCalculation
    # named OC20, but the calculator resolved to omat_pbe (the 2026-10 hub incident)
    c = AdsorptionCalculation([SumCalculator([FakeMACE("omat_pbe"), D3Calculator()])] * 3,
                              mlip_name="MACE-MH-1-OC20", benchmark="B")
    info = c._calculator_info()
    assert info["class"] == "FakeMACE" and info["head"] == "omat_pbe"
    assert "oc20_usemppbe" in info["available_heads"]
    assert "warning" not in info                    # D3 layer is not a model component
    uma = AdsorptionCalculation([FakeUMA("oc20")], mlip_name="U", benchmark="B")._calculator_info()
    assert uma == {"class": "FakeUMA", "task_name": "oc20"}
    assert AdsorptionCalculation([EMT()], mlip_name="E", benchmark="B")._calculator_info() == {"class": "EMT"}
    mixed = AdsorptionCalculation([FakeMACE("omat_pbe"), FakeMACE("oc20_usemppbe")], mlip_name="M",
                                  benchmark="B")._calculator_info()
    assert "warning" in mixed


def test_differing_calculator_lists_do_not_collide():
    """review 1.1.6 HIGH: [a, b] and [a, c] must give different stamps and cache signatures."""
    from catbench.adsorption import AdsorptionCalculation
    ab = AdsorptionCalculation([FakeMACE("omat_pbe"), FakeMACE("oc20_usemppbe")], mlip_name="M", benchmark="B")
    ac = AdsorptionCalculation([FakeMACE("omat_pbe"), FakeMACE("matpes_r2scan")], mlip_name="M", benchmark="B")
    assert ab._calculator_info() != ac._calculator_info()
    assert ab._relax_sig() != ac._relax_sig()
    assert [p["head"] for p in ab._calculator_info()["per_calculator"]] == ["omat_pbe", "oc20_usemppbe"]


def test_structure_cache_is_not_shared_across_heads():
    from catbench.adsorption import AdsorptionCalculation
    a = AdsorptionCalculation([FakeMACE("omat_pbe")], mlip_name="M", benchmark="B")
    b = AdsorptionCalculation([FakeMACE("oc20_usemppbe")], mlip_name="M", benchmark="B")
    assert a._relax_sig() != b._relax_sig()


def _old_result(d, settings):
    d.mkdir(parents=True, exist_ok=True)
    (d / "M_result.json").write_text(json.dumps({"calculation_settings": settings, "r1": {"final": {}}}))
    for name in ("M_gases.json", "M_gases_single_point.json"):
        (d / name).write_text("{}")


def test_resume_refuses_a_different_head(tmp_path):
    from catbench.adsorption import AdsorptionCalculation
    d = tmp_path / "result" / "M"
    run = AdsorptionCalculation([FakeMACE("oc20_usemppbe")], mlip_name="M", benchmark="B")
    _old_result(d, {"dispersion": {"method": "none"},
                    "calculator": {"class": "FakeMACE", "head": "omat_pbe",
                                   "available_heads": ["matpes_r2scan", "oc20_usemppbe", "omat_pbe"]}})
    with pytest.raises(RuntimeError, match="calculator"):
        run._load_existing_results(str(d))
    _old_result(d, {"dispersion": {"method": "none"}, "calculator": run._calculator_info()})
    final, *_ = run._load_existing_results(str(d))           # same head: resumes
    assert "r1" in final


def test_resume_of_pre_116_result_is_allowed_with_a_note(tmp_path, capsys):
    from catbench.adsorption import AdsorptionCalculation
    d = tmp_path / "result" / "M"
    _old_result(d, {"dispersion": {"method": "none"}})
    run = AdsorptionCalculation([FakeMACE("omat_pbe")], mlip_name="M", benchmark="B")
    old_sig = run._relax_sig()[:5]
    (d / "M_structure_cache.json").write_text(json.dumps({"__relax_config__": old_sig, "k": 1}))
    final, *_, cache = run._load_existing_results(str(d))
    assert "r1" in final and cache.get("k") == 1       # 1.1.5 run resumes, slabs not re-relaxed
    assert "no calculator stamp" in capsys.readouterr().out
    # review 1.1.6 MEDIUM: the new stamp must not vouch for the inherited reaction ...
    stamp = run._run_metadata()["calculator"]
    assert stamp["unverified_prior_reactions"] == 1 and stamp["head"] == "omat_pbe"
    # ... and the mark survives the next resume of the file this run writes
    _old_result(d, {"dispersion": {"method": "none"}, "calculator": stamp})
    again = AdsorptionCalculation([FakeMACE("omat_pbe")], mlip_name="M", benchmark="B")
    again._load_existing_results(str(d))
    assert again._run_metadata()["calculator"]["unverified_prior_reactions"] == 1
    assert "unverified" in __import__("catbench.adsorption", fromlist=["AdsorptionAnalysis"]) \
        .AdsorptionAnalysis._calculator_label(stamp)


def test_failed_only_result_is_still_guarded(tmp_path):
    """review 1.1.6 HIGH: a file holding only failures + gas energies must not be resumed
    with another head (its gas energies would be reused)."""
    from catbench.adsorption import AdsorptionCalculation
    d = tmp_path / "result" / "M"
    d.mkdir(parents=True)
    (d / "M_result.json").write_text(json.dumps({
        "calculation_settings": {"dispersion": {"method": "none"},
                                 "calculator": {"class": "FakeMACE", "head": "omat_pbe"}},
        "_failures": {"r1": {"error": "x", "stage": "relax:adslab"}}}))
    (d / "M_gases.json").write_text(json.dumps({"H2gas": -6.7}))
    (d / "M_gases_single_point.json").write_text("{}")
    run = AdsorptionCalculation([FakeMACE("oc20_usemppbe")], mlip_name="M", benchmark="B")
    with pytest.raises(RuntimeError, match="calculator"):
        run._load_existing_results(str(d))


def test_reaction_without_clean_slab_fails_with_a_reason_and_coverage_shows_calculator(tmp_path, monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    from catbench.adsorption import AdsorptionCalculation, AdsorptionAnalysis
    monkeypatch.chdir(tmp_path)
    _dataset(str(tmp_path))                  # 3 normal reactions + "broken" without 'star'
    AdsorptionCalculation([EMT(), EMT(), EMT()], mlip_name="EMT", benchmark="COV",
                          save_files=False, n_crit_relax=60).run()
    res = json.load(open(tmp_path / "result" / "EMT" / "EMT_result.json"))
    f = res["_failures"]["broken"]
    assert f["stage"] == "input" and "clean-slab" in f["error"]
    assert res["calculation_settings"]["calculator"] == {"class": "EMT"}

    AdsorptionAnalysis().analysis()
    xlsx = [x for x in os.listdir(tmp_path) if x.endswith("_Benchmarking_Analysis.xlsx")][0]
    rows = list(openpyxl.load_workbook(tmp_path / xlsx, data_only=True)["Coverage"].iter_rows(values_only=True))
    got = dict(zip(rows[0], rows[1]))
    assert got["calculator"] == "EMT" and got["Failed_by_stage"] == "input: 1"


def test_calculator_label():
    from catbench.adsorption import AdsorptionAnalysis
    lab = AdsorptionAnalysis._calculator_label
    assert lab({"class": "MACECalculator", "head": "omat_pbe", "available_heads": ["a", "b"]}) == \
        "MACECalculator head=omat_pbe"
    assert lab(None) == "unknown"
    assert "WARNING" in lab({"class": "A", "warning": "calculators differ in model/head/task"})
