"""
Custom Relative Energy Calculation Module for CatBench.

This module provides a specialized interface for custom relative energy calculations
using the general RelativeEnergyCalculation class.
"""

from catbench.relative.calculation import RelativeEnergyCalculation


class CustomCalculation:
    """
    Custom Relative Energy Calculation wrapper for MLIP benchmarking.
    
    This class provides a custom reaction-specific interface to the general
    RelativeEnergyCalculation class, automatically setting the task_type to "custom".
    
    Args:
        calculator: ASE-compatible MLIP calculator for benchmarking
        benchmark (str): Name of the benchmark dataset
        mlip_name (str): Name identifier for the MLIP being benchmarked
        **kwargs: Additional configuration parameters
    """
    
    def __new__(cls, calculator, benchmark, mlip_name, **kwargs):
        """Create a RelativeEnergyCalculation instance with task_type='custom'."""
        return RelativeEnergyCalculation(
            calculator=calculator,
            task_type="custom",
            benchmark=benchmark,
            mlip_name=mlip_name,
            **kwargs
        )


# Convenience function (alternative to class)
def create_custom_calculation(calculator, benchmark, mlip_name, **kwargs):
    """
    Create a custom relative energy calculation instance.
    
    This is a convenience function that creates a RelativeEnergyCalculation
    configured for custom relative energy calculations.
    
    Args:
        calculator: ASE-compatible MLIP calculator
        benchmark (str): Benchmark dataset name
        mlip_name (str): MLIP identifier
        **kwargs: Additional configuration
        
    Returns:
        RelativeEnergyCalculation: Configured for custom reactions
    """
    return RelativeEnergyCalculation(
        calculator=calculator,
        task_type="custom",
        benchmark=benchmark,
        mlip_name=mlip_name,
        **kwargs
    )