"""
Bulk Formation Energy Calculation Module for CatBench.

This module provides a specialized interface for bulk formation energy calculations
using the general RelativeEnergyCalculation class.
"""

from catbench.relative.calculation import RelativeEnergyCalculation


class BulkFormationCalculation:
    """
    Bulk Formation Energy Calculation wrapper for MLIP benchmarking.
    
    This class provides a bulk formation-specific interface to the general
    RelativeEnergyCalculation class, automatically setting the task_type to "bulk_formation".
    
    Args:
        calculator: ASE-compatible MLIP calculator for benchmarking
        benchmark (str): Name of the benchmark dataset
        mlip_name (str): Name identifier for the MLIP being benchmarked
        **kwargs: Additional configuration parameters
    """
    
    def __new__(cls, calculator, benchmark, mlip_name, **kwargs):
        """Create a RelativeEnergyCalculation instance with task_type='bulk_formation'."""
        return RelativeEnergyCalculation(
            calculator=calculator,
            task_type="bulk_formation",
            benchmark=benchmark,
            mlip_name=mlip_name,
            **kwargs
        )


# Convenience function (alternative to class)
def create_bulk_formation_calculation(calculator, benchmark, mlip_name, **kwargs):
    """
    Create a bulk formation energy calculation instance.
    
    This is a convenience function that creates a RelativeEnergyCalculation
    configured for bulk formation energy calculations.
    
    Args:
        calculator: ASE-compatible MLIP calculator
        benchmark (str): Benchmark dataset name
        mlip_name (str): MLIP identifier
        **kwargs: Additional configuration
        
    Returns:
        RelativeEnergyCalculation: Configured for bulk formation energy
    """
    return RelativeEnergyCalculation(
        calculator=calculator,
        task_type="bulk_formation",
        benchmark=benchmark,
        mlip_name=mlip_name,
        **kwargs
    )