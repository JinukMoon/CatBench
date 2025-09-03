"""
Configuration presets for dispersion correction.

This module provides predefined configurations for common D3 dispersion
correction scenarios used in CatBench benchmarking.
"""

from typing import Dict, Any

# Default D3 configurations for common use cases
DISPERSION_CONFIGS = {
    'default': {
        'damping_type': 'damp_bj',
        'functional_name': 'pbe',
        'vdw_cutoff': 9000,  # au², corresponds to 50.2 Å (VASP default)
        'cn_cutoff': 1600    # au², corresponds to 21.2 Å (VASP default)
    },
    'zero_damping': {
        'damping_type': 'damp_zero',
        'functional_name': 'pbe',
        'vdw_cutoff': 9000,
        'cn_cutoff': 1600
    },
    'scan': {
        'damping_type': 'damp_bj',
        'functional_name': 'scan',
        'vdw_cutoff': 9000,
        'cn_cutoff': 1600
    },
    'scan_zero': {
        'damping_type': 'damp_zero',
        'functional_name': 'scan',
        'vdw_cutoff': 9000,
        'cn_cutoff': 1600
    },
    'b3lyp': {
        'damping_type': 'damp_bj',
        'functional_name': 'b3lyp',
        'vdw_cutoff': 9000,
        'cn_cutoff': 1600
    },
    'hse06': {
        'damping_type': 'damp_bj',
        'functional_name': 'hse06',
        'vdw_cutoff': 9000,
        'cn_cutoff': 1600
    },
    'tpss': {
        'damping_type': 'damp_bj',
        'functional_name': 'tpss',
        'vdw_cutoff': 9000,
        'cn_cutoff': 1600
    },
    # Extended cutoff configurations for large systems
    'large_system': {
        'damping_type': 'damp_bj',
        'functional_name': 'pbe',
        'vdw_cutoff': 12000,  # ~63 Å
        'cn_cutoff': 2000     # ~24 Å
    },
    # Conservative settings with tighter cutoffs
    'conservative': {
        'damping_type': 'damp_bj',
        'functional_name': 'pbe',
        'vdw_cutoff': 7000,   # ~39 Å
        'cn_cutoff': 1400     # ~19 Å
    }
}


def get_dispersion_config(preset: str = 'default') -> Dict[str, Any]:
    """
    Get predefined dispersion correction configuration.
    
    Args:
        preset: Name of the preset configuration. Available options:
            - 'default': BJ damping with PBE (recommended for most MLIPs)
            - 'zero_damping': Zero damping with PBE
            - 'scan': BJ damping with SCAN functional
            - 'scan_zero': Zero damping with SCAN functional
            - 'b3lyp': BJ damping with B3LYP functional
            - 'hse06': BJ damping with HSE06 functional
            - 'tpss': BJ damping with TPSS functional
            - 'large_system': Extended cutoffs for large systems
            - 'conservative': Reduced cutoffs for faster calculations
            
    Returns:
        dict: Dispersion correction configuration dictionary
        
    Raises:
        ValueError: If preset name is not recognized
    """
    if preset not in DISPERSION_CONFIGS:
        available = ', '.join(DISPERSION_CONFIGS.keys())
        raise ValueError(
            f"Unknown preset '{preset}'. Available presets: {available}"
        )
    
    return DISPERSION_CONFIGS[preset].copy()
