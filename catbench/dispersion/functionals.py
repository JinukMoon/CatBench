"""
Functional names the D3 kernel (cuda/pair_e3gnn/pair_d3_for_ase.cu) has damping parameters for.

The kernel looks the name up in a per-damping table. An unknown name used to print
"Error: Functional name unknown" and carry on with unset parameters, giving arbitrary D3
energies (e.g. -11.7 eV for H2 instead of -0.0025 eV). The name is therefore checked here,
in Python, before the kernel is touched -- this also protects installations whose compiled
pair_d3.so predates the kernel-side fix. tests/test_d3_functionals.py keeps these tuples
equal to the kernel tables.
"""
import difflib

D3_FUNCTIONALS = {
    "damp_zero": (
        "b-lyp", "b-p", "b1b95", "b2-plyp", "b2gp-plyp", "b3-lyp", "b3pw91", "b97-d", "bh-lyp",
        "bmk", "bop", "bpbe", "cam-b3lyp", "hcth120", "hf", "hse06", "lc-wpbe", "m05", "m052x",
        "m06", "m062x", "m06hf", "m06l", "mpw1b95", "mpwb1k", "mpwlyp", "o-lyp", "o-pbe", "otpss",
        "pbe", "pbe0", "pbe38", "pbesol", "ptpss", "pw6b95", "pwb6k", "pwpb95", "revpbe",
        "revpbe0", "revpbe38", "revssb", "rpbe", "rpw86-pbe", "slater-dirac-exchange", "ssb",
        "tpss", "tpss0", "tpssh",
    ),
    "damp_bj": (
        "b-lyp", "b-p", "b1b95", "b2-plyp", "b2gp-plyp", "b3-lyp", "b3-lyp/6-31gd", "b3pw91",
        "b97-d", "bh-lyp", "bmk", "bop", "bpbe", "cam-b3lyp", "dsd-blyp", "dsd-blyp-fc", "hcth120",
        "hf", "hf/minis", "hf/mixed", "hf/sv", "hf3c", "hf3cv", "hse06", "hsesol", "lc-wpbe",
        "mpw1b95", "mpwlyp", "o-lyp", "opbe", "otpss", "pbe", "pbe0", "pbeh-3c", "pbeh3c",
        "pbesol", "ptpss", "pw1pw", "pw6b95", "pwb6k", "pwgga", "pwpb95", "revpbe", "revpbe0",
        "revpbe38", "revssb", "rpbe", "rpw86-pbe", "ssb", "tpss", "tpss0", "tpssh", "wb97m",
    ),
}

# Not in this kernel's tables. Named so the error can say so instead of only "unknown".
_KNOWN_MISSING = {"scan", "r2scan", "r2scan-3c", "rscan"}


def check_d3_functional(damping_type, functional_name):
    """Return the lower-cased functional name, or raise ValueError if the D3 kernel has no
    parameters for it under this damping (instead of computing with unset parameters)."""
    damping = str(damping_type).lower()
    name = str(functional_name).lower()
    if damping not in D3_FUNCTIONALS:
        raise ValueError(f"Unknown D3 damping type '{damping_type}'. Use one of {sorted(D3_FUNCTIONALS)}.")
    supported = D3_FUNCTIONALS[damping]
    if name in supported:
        return name
    msg = f"D3 has no '{damping}' parameters for functional '{functional_name}'."
    if name in _KNOWN_MISSING:
        msg += (" SCAN-family D3 parameters are not part of this D3 implementation; use a D3 code "
                "that has them (e.g. simple-dftd3 / the `dftd3` package for r2SCAN-D3(BJ)).")
    close = difflib.get_close_matches(name, supported, n=3, cutoff=0.6)
    if close:
        msg += f" Did you mean {', '.join(repr(c) for c in close)}?"
    raise ValueError(msg + f" Supported names: {', '.join(supported)}.")
