"""
Dispersion correction implementation for CatBench.

This module provides GPU-accelerated Grimme's D3 dispersion correction
that can be combined with any ASE calculator for enhanced accuracy.

References:
    - Grimme et al., J. Chem. Phys. 132, 154104 (2010)
    - SevenNet: Park et al., J. Chem. Theory Comput. 20, 4857 (2024)
"""

import sys
import os
from typing import List, Dict, Any, Optional, Union
from ase.calculators.calculator import Calculator
from ase.calculators.mixing import SumCalculator

# Add cuda path for D3Calculator import
_d3_path = os.path.join(os.path.dirname(__file__), 'cuda')
if _d3_path not in sys.path:
    sys.path.insert(0, _d3_path)


class DispersionCorrection:
    """
    Dispersion correction class for applying D3 corrections to MLIP calculators.
    
    This class provides methods to apply Grimme's D3 dispersion correction to
    ASE calculators, improving their accuracy for van der Waals interactions.
    
    The correction is GPU-accelerated using CUDA and automatically compiles
    required kernels on first use.
    
    Attributes:
        damping_type (str): Type of damping function used
        functional_name (str): DFT functional for D3 parameters
        vdw_cutoff (float): van der Waals cutoff distance
        cn_cutoff (float): Coordination number cutoff distance
    """
    
    def __init__(
        self,
        damping_type: str = 'damp_bj',
        functional_name: str = 'pbe',
        vdw_cutoff: float = 9000,
        cn_cutoff: float = 1600,
        **kwargs
    ):
        """
        Initialize dispersion correction with specified parameters.
        
        Args:
            damping_type: Type of damping function. Options:
                - 'damp_bj': Becke-Johnson damping (recommended)
                - 'damp_zero': Zero damping
            functional_name: DFT functional for D3 parameters. Options include:
                - 'pbe': Perdew-Burke-Ernzerhof (default, most MLIPs)
                - 'scan': Strongly constrained and appropriately normed
                - 'b3lyp': B3LYP hybrid functional
                - 'hse06': Heyd-Scuseria-Ernzerhof
                - See VASP documentation for full list
            vdw_cutoff: van der Waals cutoff in au² (default: 9000 = 50.2 Å)
            cn_cutoff: Coordination number cutoff in au² (default: 1600 = 21.2 Å)
            **kwargs: Additional parameters passed to underlying D3Calculator
            
        Raises:
            ImportError: If D3Calculator cannot be imported
            RuntimeError: If CUDA is not available
        """
        self.damping_type = damping_type
        self.functional_name = functional_name
        self.vdw_cutoff = vdw_cutoff
        self.cn_cutoff = cn_cutoff
        self.kwargs = kwargs
        
        # Try to import D3Calculator
        self._d3_calculator_class = self._get_d3_calculator_class()
        
        # Create and cache D3 calculator instance
        self._d3_calc = None
        
    def _get_d3_calculator_class(self):
        """Get the D3Calculator class, trying multiple import paths."""
        try:
            # Check if torch is available first
            import torch
        except ImportError:
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
        
        try:
            # Import D3Calculator directly (path already added to sys.path)
            from d3_calculator import D3Calculator
            return D3Calculator
        except ImportError as e:
            raise ImportError(
                "Failed to import D3Calculator. Make sure CUDA toolkit and "
                "required dependencies are installed.\n"
                f"Error: {e}"
            )
    
    def _get_d3_calculator(self):
        """Get or create the D3 calculator instance."""
        if self._d3_calc is None:
            self._d3_calc = self._d3_calculator_class(
                damping_type=self.damping_type,
                functional_name=self.functional_name,
                vdw_cutoff=self.vdw_cutoff,
                cn_cutoff=self.cn_cutoff,
                **self.kwargs
            )
        return self._d3_calc
    
    def apply(self, calculator: Calculator) -> SumCalculator:
        """
        Apply D3 dispersion correction to a calculator.
        
        Args:
            calculator: ASE-compatible calculator to apply correction to
            
        Returns:
            SumCalculator: Combined calculator with D3 correction
        """
        d3_calc = self._get_d3_calculator()
        return SumCalculator([calculator, d3_calc])
    
    
    @classmethod
    def from_config(cls, config: Union[str, Dict[str, Any]]) -> 'DispersionCorrection':
        """
        Create DispersionCorrection from configuration.
        
        Args:
            config: Either a string preset name or configuration dictionary.
                Preset options: 'default', 'zero_damping', 'scan', 'b3lyp'
                
        Returns:
            DispersionCorrection: Configured dispersion correction instance
        """
        if isinstance(config, str):
            from catbench.dispersion.config import get_dispersion_config
            config = get_dispersion_config(config)
        
        return cls(**config)
    
    def __repr__(self) -> str:
        """String representation of DispersionCorrection."""
        return (
            f"DispersionCorrection("
            f"damping='{self.damping_type}', "
            f"functional='{self.functional_name}', "
            f"vdw_cutoff={self.vdw_cutoff}, "
            f"cn_cutoff={self.cn_cutoff})"
        )


# Convenience functions for quick usage
def apply_d3_correction(
    calculator: Calculator,
    **kwargs
) -> SumCalculator:
    """
    Convenience function to apply D3 correction to a calculator.
    
    This is a shorthand for creating a DispersionCorrection instance
    and applying it to a calculator.
    
    Args:
        calculator: ASE calculator to apply correction to
        **kwargs: Parameters for DispersionCorrection
        
    Returns:
        SumCalculator: Calculator with D3 correction
    """
    d3_corr = DispersionCorrection(**kwargs)
    return d3_corr.apply(calculator)