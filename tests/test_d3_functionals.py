"""D3 functional names (1.1.7): an unknown name must raise instead of computing with unset
damping parameters (it gave -11.7 eV for H2 instead of -0.0025 eV)."""
import os
import re

import pytest

from catbench.dispersion.config import DISPERSION_CONFIGS, get_dispersion_config
from catbench.dispersion.functionals import D3_FUNCTIONALS, check_d3_functional

CU = os.path.join(os.path.dirname(os.path.dirname(__file__)), "catbench", "dispersion", "cuda",
                  "pair_e3gnn", "pair_d3_for_ase.cu")
TABLE_FN = {"damp_zero": "setfuncpar_zero", "damp_bj": "setfuncpar_bj"}


def _kernel_table(damping):
    src = open(CU).read()
    body = src.split(f"void PairD3::{TABLE_FN[damping]}()")[1].split("\nvoid PairD3::")[0]
    codes = {n: int(c) for n, c in re.findall(r'\{\s*"([^"]+)"\s*,\s*(\d+)\s*\}', body)}
    cases = {}
    for c, stmt in re.findall(r'case\s+(\d+)\s*:(.*?)\bbreak;', body, re.S):
        cases[int(c)] = {k: float(v) for k, v in re.findall(r'(rs6|s18|rs18|s6)\s*=\s*(-?[\d.]+)', stmt)}
    return codes, cases


def _params(damping, name):
    codes, cases = _kernel_table(damping)
    return cases[codes[name]]


@pytest.mark.parametrize("damping", sorted(D3_FUNCTIONALS))
def test_python_names_equal_the_kernel_tables(damping):
    codes, cases = _kernel_table(damping)
    assert set(D3_FUNCTIONALS[damping]) == {n for n, c in codes.items() if c in cases}


@pytest.mark.parametrize("name", ["r2scan", "scan", "b3lyp", "not-a-functional"])
def test_unknown_functional_raises_before_any_calculation(name):
    from catbench.dispersion import DispersionCorrection
    with pytest.raises(ValueError) as e:
        DispersionCorrection(functional_name=name)        # no torch/GPU needed to refuse
    msg = str(e.value)
    if "scan" in name:
        assert "SCAN" in msg
    if name == "b3lyp":
        assert "'b3-lyp'" in msg                          # points to the table spelling


def test_known_names_pass_case_insensitive():
    assert check_d3_functional("damp_bj", "PBE") == "pbe"
    assert check_d3_functional("damp_zero", "rpbe") == "rpbe"
    with pytest.raises(ValueError):
        check_d3_functional("damp_bjm", "pbe")             # only damp_bj / damp_zero are wired


def test_every_preset_is_usable_and_scan_presets_are_gone():
    for name, cfg in DISPERSION_CONFIGS.items():
        check_d3_functional(cfg["damping_type"], cfg["functional_name"])
    with pytest.raises(ValueError, match="removed"):
        get_dispersion_config("scan")


@pytest.mark.parametrize("name,expected", [
    ("pbe", {"rs6": 0.4289, "s18": 0.7875, "rs18": 4.4407}),
    ("rpbe", {"rs6": 0.1820, "s18": 0.8318, "rs18": 4.0094}),
    ("revpbe", {"rs6": 0.5238, "s18": 2.3550, "rs18": 3.5016}),
    ("pbesol", {"rs6": 0.4466, "s18": 2.9491, "rs18": 6.1742}),
    ("b3-lyp", {"rs6": 0.3981, "s18": 1.9889, "rs18": 4.4211}),
    ("b2-plyp", {"rs6": 0.3065, "s18": 0.9147, "rs18": 5.0570, "s6": 0.64}),   # s6 was dead code
])
def test_bj_parameters(name, expected):
    """Grimme D3(BJ) values (a1=rs6, s8=s18, a2=rs18); pbe & co. must not change."""
    assert _params("damp_bj", name) == expected


def test_kernel_error_is_fatal_not_a_print():
    hdr = open(CU.replace(".cu", ".h")).read()
    body = hdr.split("class Error")[1].split("};")[0]
    assert "abort()" in body


def test_pbe_d3_energies_unchanged_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from ase.build import add_adsorbate, fcc111, molecule
    from catbench.dispersion.cuda.d3_calculator import D3Calculator
    h2 = molecule("H2"); h2.center(vacuum=6.0); h2.pbc = True
    slab = fcc111("Pt", (2, 2, 3), vacuum=8.0); add_adsorbate(slab, "O", 1.2, "fcc")
    try:
        d3 = D3Calculator(damping_type="damp_bj", functional_name="pbe")
    except (RuntimeError, OSError) as e:                       # kernel not buildable here (e.g. no ninja/nvcc)
        pytest.skip(f"D3 kernel unavailable: {e}")
    for atoms, ref in ((h2, -0.002497), (slab, -6.384434)):   # 1.1.6 kernel, 2026-10-20
        atoms.calc = d3
        assert atoms.get_potential_energy() == pytest.approx(ref, abs=2e-6)
