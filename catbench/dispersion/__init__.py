"""
Dispersion correction module for CatBench.

This module provides Grimme's D3 dispersion correction for MLIPs,
adding van der Waals interactions that are often missing or poorly
described in machine learning interatomic potentials.

Note: This module requires PyTorch and CUDA. Install with:
  pip install catbench[d3]
"""

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Only import if torch is available
if _TORCH_AVAILABLE:
    from catbench.dispersion.correction import DispersionCorrection
    from catbench.dispersion.config import DISPERSION_CONFIGS, get_dispersion_config
    
    __all__ = [
        'DispersionCorrection',
        'DISPERSION_CONFIGS',
        'get_dispersion_config'
    ]
else:
    # Provide dummy classes with helpful error messages
    def _raise_import_error(*args, **kwargs):
        raise ImportError(
            "\n" + "="*60 + "\n"
            "PyTorch is required for D3 dispersion correction.\n\n"
            "To use dispersion features, please install with:\n"
            "  pip install catbench[d3]\n\n"
            "Or install PyTorch separately:\n"
            "  pip install torch\n\n"
            "Note: CUDA toolkit is also required for GPU acceleration.\n"
            + "="*60
        )
    
    class DispersionCorrection:
        def __init__(self, *args, **kwargs):
            _raise_import_error()
    
    DISPERSION_CONFIGS = {}
    get_dispersion_config = _raise_import_error
    
    __all__ = [
        'DispersionCorrection',
        'DISPERSION_CONFIGS', 
        'get_dispersion_config'
    ]