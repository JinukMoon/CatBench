"""
CatBench: A comprehensive benchmarking suite for Machine Learning Interatomic Potentials (MLIPs).

CatBench provides tools for downloading catalytic reaction data, preprocessing datasets,
running MLIP calculations, and analyzing results with comprehensive visualization and
statistical analysis capabilities.

Main modules:
    - catbench.adsorption: Adsorption energy benchmarking
    - catbench.eos: Equation of state benchmarking  
    - catbench.relative: Relative energy benchmarking (surface, bulk formation, custom)
    - catbench.dispersion: Dispersion corrections
    - catbench.utils: Utility functions

Example usage:
    # Import from specific submodules
    >>> from catbench.adsorption import AdsorptionCalculation, AdsorptionAnalysis
    >>> from catbench.eos import EOSCalculation, EOSAnalysis  
    >>> from catbench.relative import RelativeEnergyCalculation, RelativeEnergyAnalysis
    
    # Task-specific convenience classes (optional)
    >>> from catbench.relative.surface_energy import SurfaceEnergyCalculation
    >>> from catbench.relative.bulk_formation import BulkFormationCalculation
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("catbench")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = ["__version__"]