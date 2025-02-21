# CatBench
CatBench: Benchmark Framework for Machine Learning Potentials in Adsorption Energy Predictions

## Installation

```bash
pip install catbench
```

## Overview
![CatBench Schematic](assets/CatBench_Schematic.png)
CatBench is a comprehensive benchmarking framework designed to evaluate Machine Learning Potentials (MLPs) for adsorption energy predictions. It provides tools for data processing, model evaluation, and result analysis.

## Usage Workflow

### 1. Data Processing
CatBench supports two types of data sources:

#### A. Direct from Catalysis-Hub

```python
# Import the catbench package
import catbench

# Process data from Catalysis-Hub
catbench.cathub_preprocess("Catalysis-Hub_Dataset_tag")
```

**Example:**
```python
# Process specific dataset from Catalysis-Hub
# Using AraComputational2022 as an example
catbench.cathub_preprocess("AraComputational2022")
```

#### B. User Dataset
For custom datasets, prepare your data structure as follows:

The data structure should include:
- Gas references (`gas/`) containing VASP output files for gas phase molecules
- Surface structures (`surface1/`, `surface2/`, etc.) containing:
  - Clean slab calculations (`slab/`)
  - Adsorbate-surface systems (`H/`, `OH/`, etc.)

Note: Each directory must contain CONTCAR and OSZICAR files. Other VASP output files can be present as well - the `process_output` function will automatically clean up (delete) all files except CONTCAR and OSZICAR.

```
data/
├── gas/
│   ├── H2gas/
│   │   ├── CONTCAR
│   │   └── OSZICAR
│   └── H2Ogas/
│       ├── CONTCAR
│       └── OSZICAR
├── surface1/
│   ├── slab/
│   │   ├── CONTCAR
│   │   └── OSZICAR
│   ├── H/
│   │   ├── 1/
│   │   │   ├── CONTCAR
│   │   │   └── OSZICAR
│   │   └── 2/
│   │       ├── CONTCAR
│   │       └── OSZICAR
│   └── OH/
│       ├── 1/
│       │   ├── CONTCAR
│       │   └── OSZICAR
│       └── 2/
│           ├── CONTCAR
│           └── OSZICAR
└── surface2/
    ├── slab/
    │   ├── CONTCAR
    │   └── OSZICAR
    ├── H/
    │   ├── 1/
    │   │   ├── CONTCAR
    │   │   └── OSZICAR
    │   └── 2/
    │       ├── CONTCAR
    │       └── OSZICAR
    └── OH/
        ├── 1/
        │   ├── CONTCAR
        │   └── OSZICAR
        └── 2/
            ├── CONTCAR
            └── OSZICAR
```

Then process using:

```python
import catbench

# Define coefficients for calculating adsorption energies
# For each adsorbate, specify coefficients based on the reaction equation:
# Example for H*: 
#   E_ads(H*) = E(H*) - E(slab) - 1/2 E(H2_gas)
# Example for OH*:
#   E_ads(OH*) = E(OH*) - E(slab) + 1/2 E(H2_gas) - E(H2O_gas)

coeff_setting = {
    "H": {
        "slab": -1,      # Coefficient for clean surface
        "adslab": 1,     # Coefficient for adsorbate-surface system
        "H2gas": -1/2,   # Coefficient for H2 gas reference
    },
    "OH": {
        "slab": -1,      # Coefficient for clean surface
        "adslab": 1,     # Coefficient for adsorbate-surface system
        "H2gas": +1/2,   # Coefficient for H2 gas reference
        "H2Ogas": -1,    # Coefficient for H2O gas reference
    },
}

# This will clean up directories and keep only CONTCAR and OSZICAR files
catbench.process_output("data", coeff_setting)
catbench.userdata_preprocess("data")
```

### 2. Execute Benchmark

#### A. General Benchmark
This is a general benchmark setup. The `range()` value determines the number of repetitions for reproducibility testing. If reproducibility testing is not needed, it can be set to 1.

Note: This benchmark is only compatible with MLP models that output total system energy. For example, OC20 MLP models that are trained to directly predict adsorption energies cannot be used with this framework.

```python
import catbench
from your_calculator import Calculator

# Prepare calculator list
# range(5): Run 5 times for reproducibility testing
# range(1): Single run when reproducibility testing is not needed
calculators = [Calculator() for _ in range(5)]

config = {}
catbench.execute_benchmark(calculators, **config)
```

After execution, the following files and directories will be created:

1. A `result` directory is created to store all calculation outputs.
2. Inside the `result` directory, subdirectories are created for each MLP.
3. Each MLP's subdirectory contains:
   - `gases/`: Gas reference molecules for adsorption energy calculations
   - `log/`: Slab and adslab calculation logs
   - `traj/`: Slab and adslab trajectory files
   - `{MLP_name}_gases.json`: Gas molecules energies
   - `{MLP_name}_anomaly_detection.json`: Anomaly detection status for each adsorption data
   - `{MLP_name}_result.json`: Raw data (energies, calculation times, anomaly detection, slab displacements, etc.)

#### B. OC20 MLP Benchmark
Since OC20 project MLP models are trained to predict adsorption energies directly rather than total energies, they are handled with a separate function.

```python
import catbench
from your_calculator import Calculator

# Prepare calculator list
# range(5): Run 5 times for reproducibility testing
# range(1): Single run when reproducibility testing is not needed
calculators = [Calculator() for _ in range(5)]

config = {}
catbench.execute_benchmark_OC20(calculators, **config)
```

The overall usage is similar to the general benchmark, but each MLP will only have the following subdirectories:

- `log/`: Slab and adslab calculation logs
- `traj/`: Slab and adslab trajectory files
- `{MLP_name}_anomaly_detection.json`: Anomaly detection status for each adsorption data
- `{MLP_name}_result.json`: Raw data (energies, calculation times, anomaly detection, slab displacements, etc.)

#### C. Single-point Calculation Benchmark

```python
import catbench
from your_calculator import Calculator

calculator = Calculator()

config = {}
catbench.execute_benchmark_single(calculator, **config)
```

### 3. Analysis

```python
import catbench

config = {}
catbench.analysis_MLPs(**config)
```

The analysis function processes the calculation data stored in the `result` directory and generates:

1. A `plot/` directory:
   - Parity plots for each MLP model
   - Combined parity plots for comparison
   - Performance visualization plots

2. An Excel file `{dataset_name}_Benchmarking_Analysis.xlsx`:
   - Comprehensive performance metrics for all MLP models
   - Statistical analysis of predictions
   - Model-specific details and parameters

#### Single-point Calculation Analysis

```python
import catbench

config = {}
catbench.analysis_MLPs_single(**config)
```

## Outputs

### 1. Adsorption Energy Parity Plot (mono_version & multi_version)
You can plot adsorption energy parity plots for each adsorbate across all MLPs, either simply or by adsorbate.
<p float="left">
  <img src="assets/mono_plot.png" width="400" />
  <img src="assets/multi_plot.png" width="400" />
</p>

### 2. Comprehensive Performance Table
View various metrics for all MLPs.
![Comparison Table](assets/comparison_table.png)

### 3. Anomaly Analysis
See how anomalies are detected for all MLPs.
![Comparison Table](assets/anomaly_table.png)

### 4. Analysis by Adsorbate
Observe how each MLP predicts for each adsorbate.
![Comparison Table](assets/adsorbate_comp_table.png)

## Configuration Options

### execute_benchmark / execute_benchmark_OC20
| Option | Description | Default |
|--------|-------------|---------|
| MLP_name | Name of your MLP | Required |
| benchmark | Name of benchmark dataset | Required |
| F_CRIT_RELAX | Force convergence criterion | 0.05 |
| N_CRIT_RELAX | Maximum number of steps | 999 |
| rate | Fix ratio for surface atoms (0: use original constraints, >0: fix atoms from bottom up to specified ratio) | 0.5 |
| disp_thrs_slab | Displacement threshold for slab | 1.0 |
| disp_thrs_ads | Displacement threshold for adsorbate | 1.5 |
| again_seed | Seed variation threshold | 0.2 |
| damping | Damping factor for optimization | 1.0 |
| gas_distance | Cell size for gas molecules | 10 |
| optimizer | Optimization algorithm | "LBFGS" |

### execute_benchmark_single
| Option | Description | Default |
|--------|-------------|---------|
| MLP_name | Name of your MLP | Required |
| benchmark | Name of benchmark dataset | Required |
| gas_distance | Cell size for gas molecules | 10 |

### analysis_MLPs
| Option | Description | Default |
|--------|-------------|---------|
| Benchmarking_name | Name for output files | Current directory name |
| calculating_path | Path to result directory | "./result" |
| MLP_list | List of MLPs to analyze | All MLPs in result directory |
| target_adsorbates | Target adsorbates to analyze | All adsorbates |
| specific_color | Color for plots | "black" |
| min | Axis minimum | Auto-calculated |
| max | Axis maximum | Auto-calculated |
| figsize | Figure size | (9, 8) |
| mark_size | Marker size | 100 |
| linewidths | Line width | 1.5 |
| dpi | Plot resolution | 300 |
| legend_off | Toggle legend | False |
| error_bar_display | Toggle error bars | False |
| font_setting | Font setting <br> (Eg: `["/Users/user/Library/Fonts/Helvetica.ttf", "sans-serif"]`) | False |


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
This work will be published soon.