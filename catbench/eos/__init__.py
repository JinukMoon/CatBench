"""
EOS (Equation of State) module for CatBench.

This module provides tools for EOS benchmarking including:
- Data preprocessing from VASP calculations
- Single point energy calculations with MLIPs
- Analysis and visualization of EOS curves
- Comparison between VASP and multiple MLIPs
"""

from catbench.eos.data.vasp import eos_vasp_preprocessing
from catbench.eos.calculation.calculation import EOSCalculation
from catbench.eos.analysis.analysis import EOSAnalysis

__all__ = [
    "eos_vasp_preprocessing",
    "EOSCalculation", 
    "EOSAnalysis"
]