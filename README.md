# CatBench

<p align="center">
  <img src="assets/CatBench_logo.png" alt="CatBench Logo" width="600"/>
</p>

<p align="center">
  <a href="https://catbench.org"><img src="https://img.shields.io/badge/Leaderboard-catbench.org-orange.svg" alt="Leaderboard"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/catbench/"><img src="https://img.shields.io/pypi/v/catbench" alt="Version"></a>
</p>

**A benchmarking framework for Machine Learning Interatomic Potentials in heterogeneous catalysis.**

CatBench evaluates MLIPs against DFT references across four task types — adsorption energy, surface energy, bulk formation energy, and equation of state — with automated data processing, reproducible calculation workflows, and statistical anomaly detection.

![CatBench Schematic](assets/CatBench_Schematic.png)

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tutorials](#tutorials)
- [Adsorption Energy Benchmarking](#adsorption-energy-benchmarking)
  - [Data Preparation](#data-preparation)
  - [Calculation](#calculation)
  - [Analysis](#analysis)
  - [Output Files](#output-files)
- [Relative Energy Benchmarking](#relative-energy-benchmarking)
  - [Surface Energy](#surface-energy)
  - [Bulk Formation Energy](#bulk-formation-energy)
- [Equation of State (EOS) Benchmarking](#equation-of-state-eos-benchmarking)
- [Configuration Reference](#configuration-reference)
- [Citation](#citation)

## Installation

```bash
# Basic
pip install catbench

# With D3 dispersion correction (GPU required; CPU-only not currently supported)
pip install catbench[d3]

# Development install from source
git clone https://github.com/JinukMoon/CatBench.git
cd CatBench
pip install -e .
```

## Quick Start

Minimum viable benchmark in 5 lines:

```python
from catbench.adsorption import zenodo_download, AdsorptionCalculation, AdsorptionAnalysis
from your_mlip import YourCalculator

zenodo_download("BM_dataset")                                            # 445 KB
calc = YourCalculator(...)
AdsorptionCalculation([calc] * 3, mlip_name="MyMLIP", benchmark="BM_dataset").run()
AdsorptionAnalysis().analysis()                                          # parity plots + Excel
```

For an end-to-end walkthrough on the publication's main benchmark (MamunHighT2019) with MACE-MP-0, see the [tutorial notebook](#tutorials).

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JinukMoon/catbench/blob/main/tutorials/catbench_tutorial.ipynb)

**[`tutorials/catbench_tutorial.ipynb`](tutorials/catbench_tutorial.ipynb)** — complete walkthrough on a 20-reaction subset of MamunHighT2019 using MACE-MP-0 as the representative MLIP. Covers Zenodo download, calculator setup, calculation, analysis, and the multi-MLIP comparison workflow. Runs in about 10 minutes on a Colab T4 GPU.

## Adsorption Energy Benchmarking

### Data Preparation

CatBench supports three data sources, ordered by convenience.

#### Option A: Pre-formatted Zenodo Download (Fastest, Recommended)

Five main benchmark datasets from the CatBench publication are hosted as pre-processed JSON files on Zenodo ([DOI: 10.5281/zenodo.17157086](https://zenodo.org/records/17157086)):

```python
from catbench.adsorption import zenodo_download, list_zenodo_benchmarks

list_zenodo_benchmarks()
# → ['BM_dataset', 'ComerGeneralized2024', 'FG_dataset', 'KHLOHC_origin', 'MamunHighT2019']

zenodo_download("MamunHighT2019")   # writes raw_data/MamunHighT2019_adsorption.json
```

| Benchmark | Size | Description |
|---|---|---|
| MamunHighT2019       | 195 MB | 45,130 small-molecule adsorptions on 2,035 bimetallic alloys |
| FG_dataset           | 15 MB  | 2,651 C1–C10 organic molecules on transition metals |
| KHLOHC_origin        | 11 MB  | Liquid organic hydrogen carrier adsorption (fine-tuning) |
| ComerGeneralized2024 | 2 MB   | 325 adsorptions on metal oxide surfaces |
| BM_dataset           | 0.4 MB | 32 industrial large molecules (biomass, polyurethane, plastics) |

#### Option B: CatHub Database

For benchmarks not on Zenodo, download and preprocess directly from CatHub:

```python
from catbench.adsorption import cathub_preprocessing

cathub_preprocessing("MamunHighT2019")

# Multiple datasets with adsorbate name unification:
cathub_preprocessing(
    ["MamunHighT2019", "AraComputational2022"],
    adsorbate_integration={"HO": "OH", "O2H": "OOH"},
)
```

#### Option C: User VASP Data

> **Critical: Use `rate=None` when calling `AdsorptionCalculation` on your own VASP data.**
>
> The default `rate=0.5` fixes the bottom 50% of atoms by z-coordinate and ignores your VASP Selective-Dynamics (T/F) flags. For CatHub/Zenodo data this is what the reference calculations did, but for your own data it silently overrides your physics and produces energies inconsistent with your DFT references. This is the single most common "why don't my MLIP and DFT match?" pitfall.

> **Warning:** `vasp_preprocessing` deletes every file except `CONTCAR` and `OSZICAR` to save disk space. Always run it on a copy of your original VASP output.

Organize the data as follows. `gas`, `slab`, and the `<name>gas` pattern are reserved; everything else is arbitrary:

```
your_dataset_name/
├── gas/
│   ├── H2gas/     { CONTCAR, OSZICAR }    # gas-reference folder, must end with "gas"
│   └── H2Ogas/    { CONTCAR, OSZICAR }
├── system_A/
│   ├── slab/      { CONTCAR, OSZICAR }    # reserved name (clean surface)
│   ├── H/
│   │   ├── site_0/  { CONTCAR, OSZICAR }  # any name for site variants
│   │   └── site_1/  { CONTCAR, OSZICAR }
│   └── OH/
│       └── ...
└── system_B/
    └── ...
```

Declare the reaction stoichiometry and preprocess:

```python
from catbench.adsorption import vasp_preprocessing

coeff_setting = {
    "H":  {"slab": -1, "adslab": 1, "H2gas": -1/2},
    "OH": {"slab": -1, "adslab": 1, "H2gas": +1/2, "H2Ogas": -1},
}
vasp_preprocessing("your_dataset_name", coeff_setting)
# → raw_data/your_dataset_name_adsorption.json
```

The keys `"slab"` and `"adslab"` are required literals on every entry; all other keys are gas-phase references and must end with `"gas"`. `vasp_preprocessing` validates these rules before deleting anything.

### Calculation

`AdsorptionCalculation` takes a **list** of calculators. Running the same calculator multiple times provides reproducibility statistics that the analysis step uses for anomaly detection.

```python
from catbench.adsorption import AdsorptionCalculation
from your_mlip import YourCalculator

calc = YourCalculator(...)

AdsorptionCalculation(
    [calc] * 3,                    # 3 reproducibility seeds
    mlip_name="YourMLIP",          # free-form label — folder name under result/, display name in plots
    benchmark="dataset_name",
    # rate=None,                   # REQUIRED for user VASP data (Option C)
    # save_files=False,            # skip trajectory + log files to save disk space
).run()
```

**D3 dispersion correction** (GPU required):

```python
from catbench.dispersion import DispersionCorrection

d3 = DispersionCorrection()                     # Becke-Johnson damping + PBE by default
calc_d3 = d3.apply(YourCalculator(...))
AdsorptionCalculation([calc_d3] * 3, mlip_name="YourMLIP_D3", benchmark="dataset_name").run()
```

**OC20 mode** for MLIPs that predict adsorption energy directly:

```python
AdsorptionCalculation(
    [oc20_calc] * 3, mode="oc20", mlip_name="OC20_MLIP", benchmark="dataset_name",
).run()
```

See the [Configuration Reference](#adsorptioncalculation) for all options, and the [tutorial notebook](tutorials/catbench_tutorial.ipynb) for an end-to-end runnable example.

### Analysis

```python
from catbench.adsorption import AdsorptionAnalysis

AdsorptionAnalysis().analysis()    # auto-detects every MLIP under ./result/
```

This produces:

- **Parity plots** under `./plot/<mlip_name>/{mono,multi}/` — `mono/total.png` aggregates all reactions; `multi/total.png` colors by adsorbate.
- **Excel report** `./{cwd_name}_Benchmarking_Analysis.xlsx` with MAE, RMSE, anomaly breakdown, ADwT, AMDwT, and timings across every MLIP in `./result/`.

Every data point is classified into `Normal`, `Migration`, `Energy Anom.`, `Unphys. Relax`, or `Reprod. Fail`. Thresholds are configurable — see the [Configuration Reference](#adsorptionanalysis).

**Threshold sensitivity:**

```python
AdsorptionAnalysis().threshold_sensitivity_analysis()   # displacement + bond-length by default
```

This generates stacked-area charts showing how anomaly-classification rates change with threshold values.

### Output Files

#### Parity plots

<div align="center">
<table>
<tr>
<td><img src="assets/mono_plot.png" alt="Mono Plot" width="500"/></td>
<td><img src="assets/multi_plot.png" alt="Multi Plot" width="500"/></td>
</tr>
<tr>
<td align="center"><strong>Mono</strong> — all reactions combined</td>
<td align="center"><strong>Multi</strong> — colored by adsorbate</td>
</tr>
</table>
</div>

#### Excel report

The Excel workbook has three sheet types. Example numbers from the paper:

**Main comparison sheet** — one row per MLIP:

| MLIP | Normal (%) | Anomaly (%) | MAE_total (eV) | MAE_normal (eV) | ADwT (%) | AMDwT (%) | Time/step (ms) |
|---|---|---|---|---|---|---|---|
| MLIP_A | 77.25 | 14.39 | 1.118 | 0.316 | 77.98 | 84.71 | 125.3 |
| MLIP_B | 74.22 | 16.84 | 0.667 | 0.512 | 69.66 | 80.80 | 89.7 |
| MLIP_C | 80.18 | 13.51 | 0.917 | 0.241 | 78.97 | 86.79 | 156.8 |
| ... | ... | ... | ... | ... | ... | ... | ... |

<details>
<summary>Additional sheets (click to expand)</summary>

**Anomaly breakdown** — counts per anomaly category per MLIP:

| MLIP | Normal | Migration | Energy Anom. | Unphys. Relax | Reprod. Fail |
|---|---|---|---|---|---|
| MLIP_A | 34,869 | 3,774 | 590 | 3,845 | 2,052 |
| MLIP_B | 33,503 | 4,035 | 834 | 5,221 | 1,537 |
| ... | ... | ... | ... | ... | ... |

**Per-MLIP adsorbate sheets** — one sheet per MLIP, one row per adsorbate:

| Adsorbate | Normal | Anomaly | MAE_total (eV) | MAE_normal (eV) | ADwT (%) | AMDwT (%) |
|---|---|---|---|---|---|---|
| H | 1,247 | 89 | 0.891 | 0.234 | 89.3 | 93.4 |
| OH | 1,156 | 124 | 1.045 | 0.298 | 82.7 | 87.1 |
| ... | ... | ... | ... | ... | ... | ... |

</details>

#### Threshold sensitivity charts

<div align="center">
<table>
<tr>
<td><img src="assets/disp_thrs_sensitivity.png" alt="Displacement Threshold Sensitivity" width="500"/></td>
<td><img src="assets/bond_threshold_sensitivity.png" alt="Bond Length Threshold Sensitivity" width="500"/></td>
</tr>
<tr>
<td align="center">Displacement threshold</td>
<td align="center">Bond-length change threshold</td>
</tr>
</table>
</div>

## Relative Energy Benchmarking

Same data → calculation → analysis shape as Adsorption, but with a **single** calculator (no reproducibility seeds) and a `task_type` dispatch.

### Surface Energy

> **Warning:** Preprocessing deletes all files except `CONTCAR` and `OSZICAR`. Always work on a copy.

Layout — per material, one `bulk/` and one `slab/`:

```
your_surface_data/
├── Material_1/
│   ├── bulk/  { CONTCAR, OSZICAR }
│   └── slab/  { CONTCAR, OSZICAR }
├── Material_2/
│   └── ...
```

```python
from catbench.relative.surface_energy.data import surface_energy_vasp_preprocessing
from catbench.relative import SurfaceEnergyCalculation, RelativeEnergyAnalysis

surface_energy_vasp_preprocessing("your_surface_data")
SurfaceEnergyCalculation(calculator=calc, benchmark="your_surface_data", mlip_name="MyMLIP").run()
RelativeEnergyAnalysis(task_type="surface").analysis()
```

<div align="center">
<img src="assets/surface_parity.png" alt="Surface Energy Parity Plot" width="600"/>
</div>

The Excel report provides MAE, RMSE, and max error (J/m2) across all surfaces per MLIP.

### Bulk Formation Energy

> **Warning:** Preprocessing deletes all files except `CONTCAR` and `OSZICAR`.

Layout — `bulk_compounds/` and `elements/` side-by-side:

```
your_formation_data/
├── bulk_compounds/
│   ├── Compound_1/  { CONTCAR, OSZICAR }
│   └── Compound_2/  { CONTCAR, OSZICAR }
└── elements/
    ├── Element_A/   { CONTCAR, OSZICAR }
    ├── Element_B/   { CONTCAR, OSZICAR }
    └── Element_C/   { CONTCAR, OSZICAR }
```

```python
from catbench.relative.bulk_formation.data import bulk_formation_vasp_preprocessing
from catbench.relative import BulkFormationCalculation, RelativeEnergyAnalysis

coeff_setting = {
    "Compound_1": {"bulk": 1, "Element_A": -1, "Element_C": -1/2},
    "Compound_2": {"bulk": 1, "Element_B": -2, "Element_C": -3/2},
}
bulk_formation_vasp_preprocessing("your_formation_data", coeff_setting)
BulkFormationCalculation(calculator=calc, benchmark="your_formation_data", mlip_name="MyMLIP").run()
RelativeEnergyAnalysis(task_type="bulk_formation").analysis()
```

## Equation of State (EOS) Benchmarking

Each material has N volume-point subfolders named `0`, `1`, …, `N`:

```
your_eos_data/
├── Material_1/
│   ├── 0/   { CONTCAR, OSZICAR }    # smallest volume
│   ├── 1/   { CONTCAR, OSZICAR }
│   └── ...   (up to 10, typically)
├── Material_2/
│   └── ...
```

```python
from catbench.eos import eos_vasp_preprocessing, EOSCalculation, EOSAnalysis

eos_vasp_preprocessing("your_eos_data")
EOSCalculation(calculator=calc, mlip_name="MyMLIP", benchmark="your_eos_data").run()
EOSAnalysis().analysis()
```

<div align="center">
<img src="assets/EOS_example.png" alt="EOS Analysis Example" width="600"/>
</div>

The Excel report includes Birch-Murnaghan fits with bulk modulus (B0), equilibrium volume (V0), and derivative (B0'):

| MLIP | RMSE (eV) | MAE (eV) | VASP B0 (GPa) | MLIP B0 (GPa) | B0 Error (GPa) | VASP V0 (A^3) | MLIP V0 (A^3) | V0 Error (A^3) |
|---|---|---|---|---|---|---|---|---|
| MLIP_A | 0.634 | 0.462 | 80.53 | 102.59 | 22.06 | 475.37 | 469.42 | 5.95 |
| MLIP_B | 0.411 | 0.318 | 80.53 | 72.29 | 8.24 | 475.37 | 478.51 | 3.13 |
| MLIP_C | 0.444 | 0.350 | 80.53 | 88.02 | 7.49 | 475.37 | 470.70 | 4.67 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Configuration Reference

Options are grouped into **Required**, **Commonly tuned**, and **Advanced** (collapsed). Required parameters must be passed at construction; the rest have sensible defaults and can be overridden as needed.

### AdsorptionCalculation

**Required**

| Parameter | Description | Type |
|---|---|---|
| `mlip_name` | Free-form label. Used as the folder name under `result/` and as the display name in plots and Excel sheets. | str |
| `benchmark` | Dataset name; matches `raw_data/{benchmark}_adsorption.json`. | str |

**Commonly tuned**

| Parameter | Description | Default |
|---|---|---|
| `rate` | Fraction of atoms to fix by z-coordinate. **Must be `None` for user VASP data** — see [Option C](#option-c-user-vasp-data). | 0.5 |
| `save_files` | If False, skips trajectory + log files to save disk space. | True |
| `f_crit_relax` | Force convergence criterion (eV/A). | 0.05 |
| `n_crit_relax` | Max optimization steps per structure. | 999 |
| `mode` | `"basic"` (relaxation + references) or `"oc20"` (direct E_ads prediction). | "basic" |

<details>
<summary>Advanced</summary>

| Parameter | Description | Default |
|---|---|---|
| `damping` | Optimization damping factor. | 1.0 |
| `optimizer` | ASE optimizer: LBFGS / LBFGSLineSearch / BFGS / BFGSLineSearch / GPMin / MDMin / FIRE. | "LBFGS" |
| `save_step` | Save interval for `result.json` during long runs. | 50 |
| `chemical_bond_cutoff` | Cutoff distance for bond-change detection (A). | 6.0 |

</details>

### AdsorptionAnalysis

**Commonly tuned**

| Parameter | Description | Default |
|---|---|---|
| `mlip_list` | Limit analysis to specific MLIPs. | Auto-detect all under `./result/` |
| `target_adsorbates` | Analyze only these adsorbates. | All |
| `exclude_adsorbates` | Skip these adsorbates. | None |
| `disp_thrs` | Displacement anomaly threshold (A). | 0.5 |
| `energy_thrs` | Energy anomaly threshold (eV). | 2.0 |
| `reproduction_thrs` | Cross-seed reproducibility threshold (eV). | 0.2 |
| `bond_length_change_threshold` | Bond-length-change anomaly threshold (fraction). | 0.2 |
| `energy_cutoff` | Exclude reference energies above this value (eV). | None |
| `mlip_name_map` | Display-name overrides, e.g. `{"MACE-MP-0": "MACE"}`. | {} |
| `font_setting` | `[path_to_ttf, family_name]` for custom plot font. | False |

<details>
<summary>Advanced — paths, plot styling, font sizes</summary>

| Parameter | Description | Default |
|---|---|---|
| `calculating_path` | Path to results directory. | `./result` |
| `benchmarking_name` | Output file prefix. | CWD name |
| `time_unit` | `"s"`, `"ms"`, or `"us"`. | "ms" |
| `plot_enabled` | Generate plots. | True |
| `figsize` | Figure size (width, height) in inches. | (9, 8) |
| `dpi` | Plot DPI. | 300 |
| `mark_size` | Marker size. | 100 |
| `linewidths` | Line width. | 1.5 |
| `min`, `max` | Plot axis limits. | None |
| `tick_bins`, `tick_decimal_places` | Tick control. | 6, 1 |
| `tick_labelsize` | Tick-label font size. | 25 |
| `xlabel_fontsize`, `ylabel_fontsize` | Axis-label font sizes. | 40, 40 |
| `mae_text_fontsize` | MAE-text font size. | 30 |
| `legend_fontsize`, `comparison_legend_fontsize` | Legend font sizes. | 25, 15 |
| `threshold_xlabel_fontsize`, `threshold_ylabel_fontsize` | Threshold-plot label sizes. | 40, 40 |
| `legend_off`, `mae_text_off`, `error_bar_display` | Display toggles. | False |
| `xlabel_off`, `ylabel_off`, `grid` | Display toggles. | False |
| `specific_color` | Single-MLIP plot color. | "#2077B5" |

</details>

### DispersionCorrection

| Parameter | Description | Default |
|---|---|---|
| `damping_type` | `"damp_bj"` (Becke-Johnson, recommended) or `"damp_zero"`. | "damp_bj" |
| `functional_name` | DFT functional for D3 parameters (pbe, scan, b3lyp, hse06, ...). | "pbe" |
| `vdw_cutoff` | van der Waals cutoff (au^2). | 9000 |
| `cn_cutoff` | Coordination number cutoff (au^2). | 1600 |

### Relative energy and EOS classes

`SurfaceEnergyCalculation`, `BulkFormationCalculation`, and `EOSCalculation` all require `calculator`, `benchmark`, and `mlip_name`, and accept `f_crit_relax` and `n_crit_relax` for optimization control.

`RelativeEnergyAnalysis` requires `task_type` (`"surface"`, `"bulk_formation"`, or `"custom"`) and accepts the same plotting options as `AdsorptionAnalysis`.

<details>
<summary>EOSAnalysis advanced options</summary>

| Parameter | Description | Default |
|---|---|---|
| `calculating_path` | Results directory. | `./result` |
| `plot_path` | Plot output directory. | `./plot` |
| `benchmark` | Dataset name. | CWD name |
| `mlip_list` | MLIPs to analyze. | Auto-detect |
| `figsize` | Plot dimensions. | (9, 8) |
| `dpi` | Plot DPI. | 300 |
| `mark_size` | Marker size. | 100 |
| `x_tick_bins`, `y_tick_bins` | Tick bins. | 5, 5 |
| `tick_decimal_places`, `tick_labelsize` | Tick control. | 1, 25 |
| `xlabel_fontsize`, `ylabel_fontsize` | Axis-label font sizes. | 40, 40 |
| `legend_fontsize`, `comparison_legend_fontsize` | Legend font sizes. | 25, 15 |
| `grid` | Show grid. | False |
| `font_setting` | Custom font `[path, family]`. | False |

</details>

## Citation

```bibtex
@article{catbench2025,
  title={CatBench Framework for Benchmarking Machine Learning Interatomic Potentials in Adsorption Energy Predictions for Heterogeneous Catalysis},
  author={Moon, Jinuk and Jeon, Uchan and Choung, Seokhyun and Han, Jeong Woo},
  journal={Cell Reports Physical Science},
  volume={6},
  pages={102968},
  year={2025},
  doi={10.1016/j.xcrp.2025.102968}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Contact

**Jinuk Moon** · [jumoon@snu.ac.kr](mailto:jumoon@snu.ac.kr)  
**Jeong Woo Han** · [jwhan98@snu.ac.kr](mailto:jwhan98@snu.ac.kr)  
Seoul National University

For bug reports, feature requests, and contributions: [GitHub repository](https://github.com/JinukMoon/CatBench).
