# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

CatBench is a comprehensive benchmarking framework for Machine Learning Interatomic Potentials (MLIPs) in heterogeneous catalysis. It provides automated workflows for evaluating MLIP performance against DFT references across adsorption energies, surface energies, bulk formation energies, and equation of state calculations.

## Key Components

### Package Structure
- **catbench/adsorption/**: Adsorption energy benchmarking with CatHub integration and VASP data processing
- **catbench/relative/**: Surface energy and bulk formation energy benchmarking  
- **catbench/eos/**: Equation of state benchmarking for bulk materials
- **catbench/dispersion/**: D3 dispersion correction implementation (GPU-accelerated)
- **catbench/utils/**: Shared utilities for analysis and visualization
- **catbench/config.py**: Central configuration management with all default parameters

### Data Flow Architecture
1. **Data Processing**: Converts VASP calculations or downloads CatHub data into standardized JSON format
2. **Calculation**: Runs MLIP calculations with reproducibility testing and anomaly detection
3. **Analysis**: Generates parity plots, Excel reports, and statistical metrics

## Common Development Commands

### Installation and Development Setup
```bash
# Development installation (basic features)
pip install -e .

# Development with D3 dispersion correction (requires CUDA)
pip install -e .[d3]

# Build distribution packages
python setup.py sdist bdist_wheel

# Clean build artifacts
rm -rf build/ dist/ *.egg-info/

# Upload to PyPI (maintainers only, requires credentials)
twine upload dist/*
```

### Package Building and Testing
```bash
# Install in development mode with all dependencies
pip install -e .[d3]

# Verify installation by importing core modules
python -c "from catbench.adsorption import AdsorptionCalculation, AdsorptionAnalysis; print('Import successful')"

# Check D3 dispersion availability (requires PyTorch + CUDA)
python -c "from catbench.dispersion import DispersionCorrection; print('D3 available')"

# Build package distributions for PyPI
python setup.py sdist bdist_wheel

# Publish to PyPI (automated via GitHub Actions on tag push)
# Manual publishing: twine upload dist/*
```

### Running Benchmarks
```bash
# Example workflow structure:
# 1. Data preprocessing (user VASP data or CatHub download)
# 2. MLIP calculations with multiple reproducibility runs
# 3. Analysis with parity plots and Excel reports

# Typical adsorption benchmarking workflow:
python -c "
from catbench.adsorption import vasp_preprocessing, AdsorptionCalculation, AdsorptionAnalysis
# vasp_preprocessing('path/to/vasp/data')  # Preprocesses VASP calculations
# calc = AdsorptionCalculation(calculators, mlip_name='YourMLIP', benchmark='dataset')
# calc.run()  # Runs calculations with checkpointing
# analysis = AdsorptionAnalysis()
# analysis.analysis()  # Generates plots and Excel reports
"
```

## Code Architecture

### Modular Design Pattern
Each benchmark type follows a consistent three-module pattern:
```
catbench/{benchmark_type}/
├── data/          # Data preprocessing (VASP, CatHub)
├── calculation/   # MLIP energy calculations  
└── analysis/      # Statistical analysis and visualization
```

### Calculation Workflow Pattern
All calculation modules (`AdsorptionCalculation`, `EOSCalculation`, `SurfaceEnergyCalculation`, etc.) follow this architecture:
- **Input**: List of ASE calculators for reproducibility testing (typically 3 instances)
- **Data Source**: Processes standardized JSON from `raw_data/{dataset}_*.json`
- **Output**: Results saved to `result/{mlip_name}/` with automatic checkpointing
- **Modes**: Supports basic mode (energy differences) and OC20 mode (direct predictions)
- **Optimization**: Uses LBFGS by default with configurable force convergence (0.05 eV/Å)

### Analysis Pattern
All analysis modules provide:
- **Auto-detection**: Automatically finds available MLIPs in result directories
- **Parity Plots**: Visual comparison with customizable styling via config.py
- **Excel Reports**: Comprehensive metrics (MAE, RMSE, R², timing, anomalies)
- **Anomaly Detection**: Identifies problematic calculations using configurable thresholds
- **Multi-MLIP Comparison**: Side-by-side performance evaluation

### Configuration Management
- **Central Config**: `catbench/config.py` contains CALCULATION_DEFAULTS, ANALYSIS_DEFAULTS, RELATIVE_ANALYSIS_DEFAULTS
- **Callable Defaults**: Lambda functions for dynamic path generation (e.g., current working directory)
- **Override System**: All classes accept kwargs to override specific defaults
- **Threshold Tuning**: Displacement (0.5Å), energy (2.0eV), and reproducibility (0.2eV) thresholds

### Data Processing Architecture
- **VASP Processing**: Extracts energies from OSZICAR, structures from CONTCAR
- **CatHub Integration**: Direct download and preprocessing of reaction data
- **JSON Standardization**: All data converted to consistent JSON format for cross-compatibility
- **Path Handling**: Uses absolute paths internally for reliability

### Reproducibility and Quality Assurance
- **Multiple Calculator Runs**: Typically 3 independent calculations to test numerical stability
- **Anomaly Detection Categories**:
  - Displacement anomalies (atoms moved >0.5Å during optimization)
  - Energy anomalies (unrealistic energies >2.0eV from reference)
  - Reproducibility failures (>0.2eV variance across calculator runs)
  - Bond length changes (>20% change in chemical bonds)
- **Checkpointing**: Automatic saving every 50 calculations (configurable)
- **Result Validation**: Automatic exclusion of problematic systems from statistical analysis

## Key Design Principles

1. **Calculator Agnostic**: Works with any ASE-compatible MLIP calculator
2. **Standardized I/O**: All data flows through JSON intermediates for reproducibility
3. **Configurable Thresholds**: Anomaly detection thresholds tunable for different systems
4. **Modular Architecture**: Independent data/calculation/analysis modules for each benchmark type
5. **Automatic Result Management**: Creates directories, handles paths, manages outputs automatically

## Important Warnings and Notes

### Data Safety
- **VASP Preprocessing Warning**: Functions like `process_output()` and `vasp_preprocessing()` DELETE all VASP files except CONTCAR and OSZICAR to save disk space
- **Always Work with Copies**: Never run preprocessing on original VASP data
- **Backup Recommendation**: Use `cp -r original_data/ data_for_catbench/` before processing

### System Requirements
- **D3 Dispersion**: Requires PyTorch with CUDA support (CPU-only not supported)
- **Memory Requirements**: Large systems may require significant RAM for ML models
- **Path Requirements**: All internal paths are absolute; result directories created automatically

### Development Patterns
- **Force Convergence**: Default 0.05 eV/Å using LBFGS optimizer
- **Checkpointing**: Results saved every 50 calculations by default
- **JSON Preservation**: Full calculation provenance maintained in output files
- **Excel Integration**: Comprehensive reports with summary statistics and raw data sheets

## Configuration Customization

### Common Configuration Overrides
```python
# Calculation configuration
config = {
    "f_crit_relax": 0.02,    # Tighter force convergence
    "save_step": 25,         # More frequent checkpointing  
    "optimizer": "BFGS",     # Alternative optimizer
}

# Analysis configuration
analysis_config = {
    "disp_thrs": 0.3,        # Stricter displacement threshold
    "energy_thrs": 1.5,      # Stricter energy threshold
    "figsize": (12, 10),     # Larger plots
    "dpi": 600,              # Higher resolution
}
```

### Plot Customization
The framework provides extensive plot customization through config.py defaults:
- Font sizes for all plot elements (axis labels, legends, tick labels)
- Color schemes and marker styles for multi-MLIP comparisons  
- Grid options, axis ranges, and tick formatting
- Custom MLIP display names via `mlip_name_map`

## Troubleshooting Common Issues

### Import Errors
- **D3 Module**: If `catbench.dispersion` import fails, install with `pip install catbench[d3]`
- **Missing Dependencies**: Core dependencies in requirements.txt (ase, xlsxwriter)

### Calculation Issues
- **Memory Errors**: Large systems may need memory management or calculator batching
- **Convergence Failures**: Adjust `f_crit_relax` or `n_crit_relax` parameters
- **Path Errors**: Ensure raw_data JSON files exist before running calculations

### Analysis Issues  
- **No Results Found**: Check that result directories exist and contain proper JSON files
- **Plot Generation**: Verify matplotlib backend compatibility for your system
- **Excel Export**: Ensure xlsxwriter is installed for report generation

## PyPI Publishing

CatBench is published to PyPI with automated workflows:

### Automated Publishing (Recommended)
- **GitHub Actions**: Automatically publishes to PyPI when tags are pushed
- **Workflow**: `.github/workflows/publish.yml` handles version extraction and publishing
- **Trigger**: `git tag v1.0.1 && git push origin v1.0.1` triggers automatic PyPI publication

### Manual Publishing (Maintainers Only)
```bash
# Ensure clean build state
rm -rf build/ dist/ *.egg-info/

# Build distributions
python setup.py sdist bdist_wheel

# Publish to PyPI (requires PyPI credentials)
twine upload dist/*
```