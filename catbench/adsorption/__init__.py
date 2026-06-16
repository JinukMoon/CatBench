"""
Adsorption energy benchmarking module for CatBench.

This module provides comprehensive tools for adsorption energy calculations
and analysis for Machine Learning Interatomic Potentials (MLIPs).
"""

from catbench.adsorption.calculation.calculation import AdsorptionCalculation
from catbench.adsorption.data.cathub import cathub_preprocessing
from catbench.adsorption.data.vasp import vasp_preprocessing, process_output
from catbench.adsorption.data.zenodo import zenodo_download, list_zenodo_benchmarks

__all__ = [
    'AdsorptionCalculation',
    'AdsorptionAnalysis',
    'cathub_preprocessing',
    'vasp_preprocessing',
    'process_output',
    'zenodo_download',
    'list_zenodo_benchmarks',
]


def __getattr__(name):
    if name == 'AdsorptionAnalysis':
        from catbench.adsorption.analysis.analysis import AdsorptionAnalysis
        return AdsorptionAnalysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")