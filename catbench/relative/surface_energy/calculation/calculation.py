"""
Surface Energy Calculation Module for CatBench.

This module provides a specialized interface for surface energy calculations
using the general RelativeEnergyCalculation class.
"""

from catbench.relative.calculation import RelativeEnergyCalculation


class SurfaceEnergyCalculation:
    """
    Surface Energy Calculation wrapper for MLIP benchmarking.
    
    This class provides a surface energy-specific interface to the general
    RelativeEnergyCalculation class, automatically setting the task_type to "surface".
    
    Args:
        calculator: ASE-compatible MLIP calculator for benchmarking
        benchmark (str): Name of the benchmark dataset
        mlip_name (str): Name identifier for the MLIP being benchmarked
        **kwargs: Additional configuration parameters
    """
    
    def __new__(cls, calculator, benchmark, mlip_name, **kwargs):
        """Create a RelativeEnergyCalculation instance with task_type='surface'."""
        return RelativeEnergyCalculation(
            calculator=calculator,
            task_type="surface",
            benchmark=benchmark,
            mlip_name=mlip_name,
            **kwargs
        )


# Convenience function (alternative to class)
def create_surface_energy_calculation(calculator, benchmark, mlip_name, **kwargs):
    """
    Create a surface energy calculation instance.
    
    This is a convenience function that creates a RelativeEnergyCalculation
    configured for surface energy calculations.
    
    Args:
        calculator: ASE-compatible MLIP calculator
        benchmark (str): Benchmark dataset name
        mlip_name (str): MLIP identifier
        **kwargs: Additional configuration
        
    Returns:
        RelativeEnergyCalculation: Configured for surface energy
    """
    return RelativeEnergyCalculation(
        calculator=calculator,
        task_type="surface",
        benchmark=benchmark,
        mlip_name=mlip_name,
        **kwargs
    )