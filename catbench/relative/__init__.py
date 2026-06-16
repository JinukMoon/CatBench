"""
Relative energy benchmarking module for CatBench.

This module provides tools for relative energy calculations including
surface energy, bulk formation energy, and custom relative energy tasks.

Submodules:
    - catbench.relative.surface_energy: Surface energy calculations
    - catbench.relative.bulk_formation: Bulk formation energy calculations  
    - catbench.relative.custom: Custom relative energy calculations

Example usage:
    # Core unified approach (recommended)
    >>> from catbench.relative import RelativeEnergyCalculation, RelativeEnergyAnalysis
    >>> calc = RelativeEnergyCalculation(task_type="surface", calculator, benchmark, mlip_name)
    >>> analysis = RelativeEnergyAnalysis(task_type="bulk_formation", **kwargs)
    
    # Task-specific convenience classes (alternative)
    >>> from catbench.relative.surface_energy import SurfaceEnergyCalculation
    >>> from catbench.relative.bulk_formation import BulkFormationCalculation
"""

# Core classes (work with all relative energy types)
from catbench.relative.calculation import RelativeEnergyCalculation

# Import task-specific classes for convenience
from catbench.relative.surface_energy.calculation.calculation import SurfaceEnergyCalculation
from catbench.relative.bulk_formation.calculation.calculation import BulkFormationCalculation
from catbench.relative.custom.calculation.calculation import CustomCalculation

__all__ = [
    'RelativeEnergyCalculation',
    'RelativeEnergyAnalysis',
    'SurfaceEnergyCalculation',
    'BulkFormationCalculation',
    'CustomCalculation',
]


def __getattr__(name):
    if name == 'RelativeEnergyAnalysis':
        from catbench.relative.analysis import RelativeEnergyAnalysis
        return RelativeEnergyAnalysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")