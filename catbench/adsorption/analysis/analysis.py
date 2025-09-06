"""
Adsorption energy analysis module for CatBench.

Provides comprehensive statistical analysis, anomaly detection, and professional reporting
for adsorption energy benchmarking results with Machine Learning Interatomic Potentials (MLIPs).
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import xlsxwriter
from ase.io import read

from catbench.config import ANALYSIS_DEFAULTS, get_default, PLOT_COLORS, PLOT_MARKERS
from catbench.utils.analysis_utils import (
    find_adsorbate, min_max, set_matplotlib_font, get_ads_eng_range, prepare_plot_data,
    get_calculator_keys, get_median_calculator_key
)
from catbench.utils.io_utils import save_anomaly_detection_results
class AdsorptionAnalysis:
    """
    Adsorption energy analysis class for MLIP benchmarking.

    Provides intelligent analysis of adsorption energy benchmarking results from
    Machine Learning Interatomic Potentials (MLIPs) with automatic structure 
    detection and anomaly identification.

    Features:
        - Automatic detection of calculation structure (slab vs adslab-only)
        - 5-category anomaly detection with intelligent caching
        - Statistical analysis with optimization and single-point MAE comparison
        - Advanced metrics: ADwT (OC20 standard) and AMDwT (max displacement)
        - Professional parity plots and comprehensive Excel reports
        - Threshold sensitivity analysis with visualization
        - Support for multiple adsorbate types with filtering options
        - Hybrid adsorbate detection: combines index-based (assumes appended atoms) 
          and element-based (detects additional/excess elements) methods

    Input Files:
        - {MLIP_name}_result.json: Calculation results from AdsorptionCalculation
        - {MLIP_name}_gases.json: Gas molecule energies

    Output Files:
        - PNG plots: Parity plots for MAE analysis 
        - {benchmark}_Benchmarking_Analysis.xlsx: Comprehensive Excel report with merged headers
        - {MLIP_name}_anomaly_detection.json: Anomaly detection results
        - Threshold sensitivity plots: Stacked area charts for threshold analysis
        - Anomaly detection results saved to JSON files

    Args:
        calculating_path (str, optional): Path to calculation results directory.
                                        Default: "./result"
        mlip_list (list, optional): List of MLIP names to analyze.
                                  Default: Auto-detect from directories
        target_adsorbates (list, optional): Specific adsorbates to analyze.
                                          If provided, only these will be analyzed.
        exclude_adsorbates (list, optional): Adsorbates to exclude from analysis.
                                           Ignored if target_adsorbates is provided.
        benchmarking_name (str, optional): Name for output files.
                                         Default: Working directory name
        time_unit (str, optional): Time unit for display ("s", "ms", "µs").
                                 Default: "ms"
        disp_thrs (float, optional): Substrate displacement threshold in Å. Default: 1.0
        reproduction_thrs (float, optional): Reproducibility threshold in eV. Default: 0.2
        energy_thrs (float, optional): Energy anomaly threshold in eV. Default: 2.0
        energy_cutoff (float, optional): Energy cutoff for analysis inclusion in eV. 
                                        Reactions with reference ads_eng > cutoff are excluded. Default: None (no filtering)
        bond_length_change_threshold (float, optional): Bond length change threshold for migration detection. Default: 0.2 (20%)
        figsize (tuple, optional): Figure size for plots. Default: (9, 8)
        dpi (int, optional): Resolution for saved plots. Default: 300
        min/max (float, optional): Manual axis limits for plots
        font_setting (tuple, optional): Font configuration (path, family)
        legend_fontsize (int, optional): Legend font size. Default: 20
        tick_labelsize (int, optional): Tick label size for both x and y axes. Default: 20
        grid (bool, optional): Show grid on parity plots. Default: True
        mlip_name_map (dict, optional): Mapping from raw MLIP names to display names for plots/Excel. Default: {} (no mapping)
        plot_enabled (bool, optional): Whether to generate plots. Default: True
                                     Set to False to skip plot generation during analysis.
    """

    def __init__(self, **kwargs):
        """Initialize analysis configuration."""
        self.calculating_path = kwargs.get("calculating_path", get_default("calculating_path", ANALYSIS_DEFAULTS))
        self.mlip_list = kwargs.get("mlip_list", get_default("mlip_list", ANALYSIS_DEFAULTS))
        self.target_adsorbates = kwargs.get("target_adsorbates", get_default("target_adsorbates", ANALYSIS_DEFAULTS))
        self.exclude_adsorbates = kwargs.get("exclude_adsorbates", get_default("exclude_adsorbates", ANALYSIS_DEFAULTS))
        self.benchmarking_name = kwargs.get("benchmarking_name", get_default("benchmarking_name", ANALYSIS_DEFAULTS))
        self.time_unit = kwargs.get("time_unit", get_default("time_unit", ANALYSIS_DEFAULTS))
        
        # Cache for performance optimization
        self._adwt_cache = {}
        self._amdwt_cache = {}
        self._mlip_result_cache = {}
        self._calculation_cache = {}

        # Anomaly detection thresholds
        self.disp_thrs = kwargs.get("disp_thrs", get_default("disp_thrs", ANALYSIS_DEFAULTS))
        self.energy_thrs = kwargs.get("energy_thrs", get_default("energy_thrs", ANALYSIS_DEFAULTS))
        self.energy_cutoff = kwargs.get("energy_cutoff", get_default("energy_cutoff", ANALYSIS_DEFAULTS))
        self.reproduction_thrs = kwargs.get("reproduction_thrs", get_default("reproduction_thrs", ANALYSIS_DEFAULTS))

        # Adsorbate migration detection thresholds
        self.bond_length_change_threshold = kwargs.get("bond_length_change_threshold", get_default("bond_length_change_threshold", ANALYSIS_DEFAULTS))

        # Plot settings
        self.figsize = kwargs.get("figsize", get_default("figsize", ANALYSIS_DEFAULTS))
        self.mark_size = kwargs.get("mark_size", get_default("mark_size", ANALYSIS_DEFAULTS))
        self.linewidths = kwargs.get("linewidths", get_default("linewidths", ANALYSIS_DEFAULTS))
        self.dpi = kwargs.get("dpi", get_default("dpi", ANALYSIS_DEFAULTS))
        self.legend_off = kwargs.get("legend_off", get_default("legend_off", ANALYSIS_DEFAULTS))
        self.mae_text_off = kwargs.get("mae_text_off", get_default("mae_text_off", ANALYSIS_DEFAULTS))
        self.error_bar_display = kwargs.get("error_bar_display", get_default("error_bar_display", ANALYSIS_DEFAULTS))
        self.font_setting = kwargs.get("font_setting", get_default("font_setting", ANALYSIS_DEFAULTS))
        self.specific_color = kwargs.get("specific_color", get_default("specific_color", ANALYSIS_DEFAULTS))
        self.min_value = kwargs.get("min", get_default("min", ANALYSIS_DEFAULTS))
        self.max_value = kwargs.get("max", get_default("max", ANALYSIS_DEFAULTS))
        # Axis label toggles
        self.xlabel_off = kwargs.get("xlabel_off", get_default("xlabel_off", ANALYSIS_DEFAULTS))
        self.ylabel_off = kwargs.get("ylabel_off", get_default("ylabel_off", ANALYSIS_DEFAULTS))
        # Tick control (None = auto)
        self.x_tick_bins = kwargs.get("x_tick_bins", get_default("x_tick_bins", ANALYSIS_DEFAULTS))
        self.y_tick_bins = kwargs.get("y_tick_bins", get_default("y_tick_bins", ANALYSIS_DEFAULTS))
        # Tick label decimal places (None = auto)
        self.tick_decimal_places = kwargs.get("tick_decimal_places", get_default("tick_decimal_places", ANALYSIS_DEFAULTS))
        # Font sizes
        self.legend_fontsize = kwargs.get("legend_fontsize", get_default("legend_fontsize", ANALYSIS_DEFAULTS))
        self.tick_labelsize = kwargs.get("tick_labelsize", get_default("tick_labelsize", ANALYSIS_DEFAULTS))
        self.xlabel_fontsize = kwargs.get("xlabel_fontsize", get_default("xlabel_fontsize", ANALYSIS_DEFAULTS))
        self.ylabel_fontsize = kwargs.get("ylabel_fontsize", get_default("ylabel_fontsize", ANALYSIS_DEFAULTS))
        self.mae_text_fontsize = kwargs.get("mae_text_fontsize", get_default("mae_text_fontsize", ANALYSIS_DEFAULTS))
        # Legend marker scaling (fontsize=20, marker_size=100)
        self.legend_markerscale = self.legend_fontsize / 20.0
        # Grid toggle
        self.grid = kwargs.get("grid", get_default("grid", ANALYSIS_DEFAULTS))
        # Display name mapping for MLIP names (for plots/Excel display only)
        self.mlip_name_map = kwargs.get("mlip_name_map", get_default("mlip_name_map", ANALYSIS_DEFAULTS)) or {}
        # Plot toggle
        self.plot_enabled = kwargs.get("plot_enabled", get_default("plot_enabled", ANALYSIS_DEFAULTS))
        # Plot generation configuration
        self.plot_enabled = kwargs.get("plot_enabled", get_default("plot_enabled", ANALYSIS_DEFAULTS))

        # Color and marker settings for multi plots
        self.colors = PLOT_COLORS
        self.markers = PLOT_MARKERS

    def _display_mlip_name(self, mlip_name):
        """Return mapped display name for MLIP if provided, else original name."""
        return self.mlip_name_map.get(mlip_name, mlip_name)


    def _get_analysis_adsorbates(self, all_adsorbates):
        """Determine which adsorbates to analyze based on target/exclude parameters."""
        if self.target_adsorbates is not None:
            # target_adsorbates has priority (ignore exclude_adsorbates)
            if self.exclude_adsorbates is not None:
                print("Warning: Both target_adsorbates and exclude_adsorbates provided. "
                      "Using target_adsorbates and ignoring exclude_adsorbates.")
            return list(self.target_adsorbates)  # Order will be sorted by data count later
        elif self.exclude_adsorbates is not None:
            # Only exclude_adsorbates provided
            return [ads for ads in all_adsorbates if ads not in self.exclude_adsorbates]  # Order will be sorted by data count later
        else:
            # Neither provided, use all
            return list(all_adsorbates)  # Order will be sorted by data count later

    def _setup_plot(self, mlip_name):
        """Setup plot directory and font configuration."""
        display_name = self._display_mlip_name(mlip_name)
        plot_save_path = os.path.join(os.getcwd(), "plot", display_name)
        mono_path = os.path.join(plot_save_path, "mono")
        multi_path = os.path.join(plot_save_path, "multi")

        os.makedirs(mono_path, exist_ok=True)
        os.makedirs(multi_path, exist_ok=True)

        if self.font_setting:
            set_matplotlib_font(self.font_setting[0], self.font_setting[1])

        return plot_save_path, mono_path, multi_path

    def _create_base_plot(self, min_value, max_value):
        """Create base plot configuration."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(min_value, max_value)
        ax.set_ylim(min_value, max_value)
        ax.plot([min_value, max_value], [min_value, max_value], "r-", zorder=100000)
        # Tick bins control
        if self.x_tick_bins is not None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=self.x_tick_bins))
        if self.y_tick_bins is not None:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=self.y_tick_bins))
        # Tick label formatting
        if self.tick_decimal_places is not None:
            fmt = f"%.{self.tick_decimal_places}f"
            ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
            ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        if self.grid:
            ax.grid(True)
        else:
            ax.grid(False)

        for spine in ax.spines.values():
            spine.set_linewidth(3)

        return fig, ax

    def _get_adsorbate_count(self, ads, ads_data):
        """Helper method to count total data points for an adsorbate across all categories."""
        total_count = 0
        if ads in ads_data:
            for category in ads_data[ads].values():
                if isinstance(category, dict) and "DFT" in category:
                    total_count += len(category["DFT"])
        return total_count

    def mono_plotter(self, ads_data, mlip_name, tag, min_value, max_value, mono_path, plot_data=None):
        """Create monochrome plot with optional pre-computed data."""
        fig, ax = self._create_base_plot(min_value, max_value)

        # Use pre-computed data if provided, otherwise compute
        if plot_data is not None:
            pass
        else:
            # Use dictionary lookup for efficient data access
            type_mapping = {
                "total": ["normal", "adsorbate_migration", "energy_anomaly", "unphysical_relaxation", "reproduction_failure"],
                "normal": ["normal"],
                "migration": ["adsorbate_migration"],
                "anomaly": ["energy_anomaly", "unphysical_relaxation", "reproduction_failure"],
                "single": ["all"]
            }
            types = type_mapping.get(tag, ["all"])
            plot_data = prepare_plot_data(ads_data, types)

        # Check if we have data to plot
        dft_data = plot_data["DFT"]
        mlip_data = plot_data["MLIP"]
        n_points = len(dft_data)

        if n_points == 0:
            # No data to plot - create empty plot and return 0 MAE
            MAE = 0.0
        else:
            # Single mono plot uses single color and marker
            scatter = ax.scatter(
                dft_data,
                mlip_data,
                color=self.specific_color,
                marker="o",
                s=self.mark_size,
                edgecolors="black",
                linewidths=self.linewidths,
                rasterized=True
            )

            if self.error_bar_display and "MLIP_min" in plot_data and tag != "single":
                # Vectorized error bar calculation
                yerr_minus = mlip_data - plot_data["MLIP_min"]
                yerr_plus = plot_data["MLIP_max"] - mlip_data
                ax.errorbar(
                    dft_data,
                    mlip_data,
                    yerr=[yerr_minus, yerr_plus],
                    fmt='none',
                    ecolor="black",
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                )

            # Efficient MAE calculation
            MAE = np.mean(np.abs(dft_data - mlip_data))

        display_name = self._display_mlip_name(mlip_name)
        mae_label = f"MAE-{display_name}: {MAE:.2f}"

        if not self.mae_text_off:
            ax.text(
                x=0.05, y=0.95,
                s=mae_label,
                transform=ax.transAxes,
                fontsize=self.mae_text_fontsize,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round", alpha=0.5, facecolor="white", 
                    edgecolor="black", pad=0.5
                ),
                zorder=100002,
            )

        if not self.xlabel_off:
            ax.set_xlabel("DFT (eV)", fontsize=self.xlabel_fontsize)
        if not self.ylabel_off:
            ax.set_ylabel(f"{display_name} (eV)", fontsize=self.ylabel_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=self.tick_labelsize)

        plt.tight_layout()
        plt.savefig(f"{mono_path}/{tag}.png", dpi=self.dpi)
        plt.close()

        return MAE

    def multi_plotter(self, ads_data, mlip_name, types, tag, min_value, max_value, multi_path):
        """Create multi-color plot with adsorbate-specific colors."""
        fig, ax = self._create_base_plot(min_value, max_value)

        # Sort adsorbates by total data count (descending) for consistent order
        analysis_adsorbates = sorted([ads for ads in ads_data.keys() if ads != "all"], 
                                   key=lambda ads: self._get_adsorbate_count(ads, ads_data), reverse=True)
        len_adsorbates = len(analysis_adsorbates)
        legend_width = len(max(analysis_adsorbates, key=len)) if analysis_adsorbates else 0

        error_sum = 0
        len_total = 0
        MAEs = {}
        scatter_handles = []

        # Check if this is a single calculation (no error bars)
        is_single_calc = tag == "single" or "MLIP_min" not in ads_data["all"].get(types[0], {})

        # Pre-compute plot data for each adsorbate
        plot_data_cache = {}
        for adsorbate in analysis_adsorbates:
            plot_data_cache[adsorbate] = prepare_plot_data(ads_data, types, adsorbate)

        for i, adsorbate in enumerate(analysis_adsorbates):
            plot_data = plot_data_cache[adsorbate]

            # Extract data arrays once
            dft_data = plot_data["DFT"]
            mlip_data = plot_data["MLIP"]
            n_points = len(dft_data)

            # Skip if no data for this adsorbate
            if n_points == 0:
                MAEs[adsorbate] = 0.0
                MAEs[f"len_{adsorbate}"] = 0
                continue

            # Set zorder so first adsorbates (sorted first) appear on top
            zorder_value = len_adsorbates - i

            scatter = ax.scatter(
                dft_data,
                mlip_data,
                color=self.colors[i],
                label=f"* {adsorbate}",
                marker=self.markers[i],
                s=self.mark_size,
                edgecolors="black",
                linewidths=self.linewidths,
                zorder=zorder_value,
                rasterized=True
            )

            if self.error_bar_display and not is_single_calc and "MLIP_min" in plot_data:
                # Vectorized error bar calculation
                yerr_minus = mlip_data - plot_data["MLIP_min"]
                yerr_plus = plot_data["MLIP_max"] - mlip_data
                ax.errorbar(
                    dft_data,
                    mlip_data,
                    yerr=[yerr_minus, yerr_plus],
                    fmt='none',
                    ecolor="black",
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                )

            scatter_handles.append(scatter)

            # Efficient MAE calculation
            abs_errors = np.abs(dft_data - mlip_data)
            MAEs[adsorbate] = np.mean(abs_errors)
            error_sum += np.sum(abs_errors)
            len_total += n_points
            MAEs[f"len_{adsorbate}"] = n_points

        # Calculate total MAE
        MAE_total = error_sum / len_total if len_total != 0 else 0
        MAEs["total"] = MAE_total

        display_name = self._display_mlip_name(mlip_name)
        mae_label = f"MAE-{display_name}: {MAE_total:.2f}"

        if not self.mae_text_off:
            ax.text(
                x=0.05, y=0.95,
                s=mae_label,
                transform=ax.transAxes,
                fontsize=self.mae_text_fontsize,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round", alpha=0.5, facecolor="white", 
                    edgecolor="black", pad=0.5
                ),
                zorder=100002,
            )

        if not self.xlabel_off:
            ax.set_xlabel("DFT (eV)", fontsize=self.xlabel_fontsize)
        if not self.ylabel_off:
            ax.set_ylabel(f"{display_name} (eV)", fontsize=self.ylabel_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=self.tick_labelsize)

        # Handle legend - always use consistent method
        if (legend_width < 8 and len_adsorbates < 6 or 
            legend_width < 5 and len_adsorbates < 8) and not self.legend_off:
            legend = ax.legend(
                loc="lower right",
                fontsize=self.legend_fontsize,
                ncol=(len_adsorbates // 7) + 1,
                markerscale=self.legend_markerscale,
            )
            legend.set_zorder(1001)  # Display legend on top (slightly above MAE text)
        else:
            if scatter_handles:
                fig_legend = plt.figure()
                fig_legend.legend(
                    handles=scatter_handles,
                    loc="center",
                    frameon=False,
                    ncol=(len_adsorbates // 7) + 1,
                    fontsize=self.legend_fontsize,
                    markerscale=self.legend_markerscale,
                )
                fig_legend.savefig(f"{multi_path}/legend.png", dpi=self.dpi, bbox_inches="tight")
                plt.close(fig_legend)

        plt.tight_layout()
        plt.savefig(f"{multi_path}/{tag}.png", dpi=self.dpi)
        plt.close()

        return MAEs

    def _create_excel_output(self, main_data, anomaly_data, MLIPs_data, analysis_adsorbates):
        """Create Excel output file."""
        output_file = f"{self.benchmarking_name}_Benchmarking_Analysis.xlsx"

        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            # Create custom MLIP_Data sheet with merged cells
            self._create_mlip_data_sheet(writer, main_data)

            if anomaly_data:
                # Create custom anomaly sheet with merged cells like main sheet
                self._create_anomaly_sheet(writer, anomaly_data)
                df_anomaly = pd.DataFrame(anomaly_data)  # Required for DataFrame formatting
            else:
                df_anomaly = pd.DataFrame()

            for mlip_name in sorted(MLIPs_data.keys(), key=str.lower):
                data_dict = MLIPs_data[mlip_name]
                data_tmp = []

                # Use cached mlip_result instead of reloading from file
                mlip_result = self._mlip_result_cache.get(mlip_name, {})

                display_name = self._display_mlip_name(mlip_name)
                for adsorbate in analysis_adsorbates:
                    if "normal" in data_dict and f"len_{adsorbate}" in data_dict["normal"]:
                        normal_count = data_dict["normal"][f"len_{adsorbate}"]
                        total_count = data_dict["total"][f"len_{adsorbate}"]

                        # Get anomaly breakdown for this adsorbate
                        anomaly_breakdown = self._get_adsorbate_anomaly_breakdown(data_dict.get("ads_data", {}), adsorbate)

                        # Anomaly count excludes adsorbate migration (now independent category)
                        anomaly_count = (anomaly_breakdown["reproduction_failure_count"] + 
                                       anomaly_breakdown["unphysical_relaxation_count"] + 
                                       anomaly_breakdown["energy_anomaly_count"])
                        anomaly_rate = (anomaly_count / total_count) * 100 if total_count > 0 else 0

                        # Calculate single MAE for this specific adsorbate
                        single_mae_adsorbate = data_dict.get("single_mae", {}).get(adsorbate, 0.0)

                        # Use cached ADwT and AMDwT values for this adsorbate
                        cache_key = (mlip_name, adsorbate)
                        adwt_adsorbate = self._adwt_cache.get(cache_key, 0.0)
                        amdwt_adsorbate = self._amdwt_cache.get(cache_key, 0.0)

                        data_tmp.append({
                            "Adsorbate_name": adsorbate,
                            "MAE_total": data_dict["total"].get(adsorbate, 0.0),
                            "MAE_normal": data_dict["normal"].get(adsorbate, 0.0),
                            "MAE_single": single_mae_adsorbate,
                            "ADwT": adwt_adsorbate,
                            "AMDwT": amdwt_adsorbate,
                            "Num_total": total_count,
                            "Num_normal": normal_count,
                            "Num_anomaly_total": anomaly_count,
                            "Num_reproduction_failure": anomaly_breakdown["reproduction_failure_count"],
                            "Num_unphysical_relaxation": anomaly_breakdown["unphysical_relaxation_count"],
                            "Num_adsorbate_migration": anomaly_breakdown["adsorbate_migration_count"],
                            "Num_energy_anomaly": anomaly_breakdown["energy_anomaly_count"],
                        })

                if data_tmp:
                    # Create custom MLIP sheet with merged cells like main sheet
                    self._create_mlip_sheet(writer, data_tmp, mlip_name)
                    data_df = pd.DataFrame(data_tmp)  # Required for DataFrame formatting
                else:
                    data_tmp = []

            # Apply formatting (same as core.py)
            workbook = writer.book
            header_format = workbook.add_format({
                "align": "center", 
                "valign": "vcenter",
                "bold": True
            })
            center_align = workbook.add_format({"align": "center", "valign": "vcenter"})
            number_format_0f = workbook.add_format(
                {"num_format": "#,##0", "align": "center", "valign": "vcenter"}
            )
            number_format_3f = workbook.add_format(
                {"num_format": "0.000", "align": "center", "valign": "vcenter"}
            )
            number_format_6f = workbook.add_format(
                {"num_format": "0.000000", "align": "center", "valign": "vcenter"}
            )

            column_formats = {
                # Rate columns
                "Anomaly_rate": (18, number_format_3f),
                "Reproduction_failure_rate": (25, number_format_3f),
                "Unphysical_relaxation_rate": (25, number_format_3f),
                "Adsorbate_migration_rate": (25, number_format_3f),  # Extended
                "Energy_anomaly_rate": (18, number_format_3f),  # Reduced
                # MAE columns
                "MAE_total": (15, number_format_3f),
                "MAE_normal": (17, number_format_3f),
                "MAE_single": (15, number_format_3f),
                # Metrics columns
                "ADwT": (12, number_format_3f),
                "AMDwT": (12, number_format_3f),
                # Count columns
                "Num_total": (12, number_format_0f),
                "Num_normal": (13, number_format_0f),
                "Num_reproduction_failure": (24, number_format_0f),
                "Num_unphysical_relaxation": (25, number_format_0f),
                "Num_adsorbate_migration": (22, number_format_0f),
                "Num_energy_anomaly": (18, number_format_0f),
                "Num_anomaly_total": (18, number_format_0f),
                # Column formatting for anomaly sheet
                "Anomaly rate (%)": (20, number_format_3f),
                "MAE_total (eV)": (15, number_format_3f),
                "MAE_normal (eV)": (17, number_format_3f),
                "MAE_single (eV)": (15, number_format_3f),
                "Num_anomaly": (15, number_format_0f),
                "slab_conv": (12, number_format_0f),
                "ads_conv": (12, number_format_0f),
                "slab_move": (12, number_format_0f),
                "ads_move": (12, number_format_0f),
                "slab_seed": (12, number_format_0f),
                "ads_seed": (12, number_format_0f),
                "ads_eng_seed": (12, number_format_0f),
                "Time_total (s)": (15, number_format_0f),
                "Steps_total": (15, number_format_0f),
                "Time_per_step (s)": (17, number_format_3f),
            }

            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]

                # Skip custom formatted sheets as they're already formatted
                mlip_names = self.mlip_list if self.mlip_list else []
                if sheet_name in ["MLIP_Data", "anomaly"] or sheet_name in mlip_names:
                    continue

                df = (
                    df_anomaly
                    if sheet_name == "anomaly" and anomaly_data
                    else pd.DataFrame(data_tmp) if 'data_tmp' in locals() and data_tmp else pd.DataFrame()
                )

                # Skip empty dataframes
                if df.empty:
                    continue

                for col_num, col_name in enumerate(df.columns):
                    worksheet.write(0, col_num, col_name, header_format)

                for col_num, col_name in enumerate(df.columns):
                    if col_name in column_formats:
                        width, fmt = column_formats[col_name]
                    else:
                        width = (
                            max(df[col_name].astype(str).map(len).max(), len(col_name)) + 10
                        )
                        fmt = center_align

                    worksheet.set_column(col_num, col_num, width, fmt)

                    if df[col_name].dtype == "object":
                        worksheet.set_column(col_num, col_num, width, center_align)
                    else:
                        worksheet.set_column(col_num, col_num, width, fmt)

                row_height = 20
                for row in range(len(df) + 1):
                    worksheet.set_row(row, row_height)

        print(f"Excel file '{output_file}' created successfully.")

    def _create_mlip_data_sheet(self, writer, main_data):
        """Create MLIP_Data sheet with custom merged cell structure."""
        workbook = writer.book
        worksheet = workbook.add_worksheet("MLIP_Data")

        # Define formats
        header_format = workbook.add_format({
            "align": "center", 
            "valign": "vcenter",
            "text_wrap": True,
            "bold": True
        })
        center_align = workbook.add_format({"align": "center", "valign": "vcenter"})
        bold_center_align = workbook.add_format({
            "align": "center", 
            "valign": "vcenter", 
            "bold": True
        })
        number_format_2f = workbook.add_format(
            {"num_format": "0.00", "align": "center", "valign": "vcenter"}
        )
        number_format_3f = workbook.add_format(
            {"num_format": "0.000", "align": "center", "valign": "vcenter"}
        )
        number_format_0f = workbook.add_format(
            {"num_format": "#,##0", "align": "center", "valign": "vcenter"}
        )

        # Column structure:
        # 0: MLIP_name, 1: Normal rate, 2: Adsorbate migration rate, 3-6: Anomaly rate (4 columns), 7: MAE_total, 8: MAE_normal, 9: MAE_single
        # 10: ADwT, 11: AMDwT, 12: Num_total, 13: Time_total, 14: Time_per_step, 15: Steps_total

        # Write main headers (Row 0)
        worksheet.merge_range(0, 0, 1, 0, "MLIP_name", header_format)  # 2 rows
        worksheet.merge_range(0, 1, 1, 1, "Normal rate (%)", header_format)  # 2 rows  
        worksheet.merge_range(0, 2, 1, 2, "Adsorbate migration rate (%)", header_format)  # 2 rows - independent
        worksheet.merge_range(0, 3, 0, 6, "Anomaly rate (%)", header_format)  # 4 columns
        worksheet.merge_range(0, 7, 1, 7, "MAE_total (eV)", header_format)  # 2 rows
        worksheet.merge_range(0, 8, 1, 8, "MAE_normal (eV)", header_format)  # 2 rows
        worksheet.merge_range(0, 9, 1, 9, "MAE_single (eV)", header_format)  # 2 rows
        worksheet.merge_range(0, 10, 1, 10, "ADwT (%)", header_format)  # 2 rows
        worksheet.merge_range(0, 11, 1, 11, "AMDwT (%)", header_format)  # 2 rows
        worksheet.merge_range(0, 12, 1, 12, "Num_total", header_format)  # 2 rows
        worksheet.merge_range(0, 13, 1, 13, "Time_total (s)", header_format)  # 2 rows
        worksheet.merge_range(0, 14, 1, 14, "Time_per_step (s)", header_format)  # 2 rows - fixed in seconds
        worksheet.merge_range(0, 15, 1, 15, "Steps_total", header_format)  # 2 rows

        # Write anomaly sub-headers (Row 1) - excluding adsorbate migration
        anomaly_subheaders = [
            "total",
            "reproduction failure",
            "unphysical relaxation", 
            "energy anomaly"
        ]

        for col, subheader in enumerate(anomaly_subheaders, 3):  # Start from column 3
            worksheet.write(1, col, subheader, header_format)

        # Write data starting from row 2
        for row, data in enumerate(main_data, 2):
            col = 0

            # MLIP name - bold formatting
            worksheet.write(row, col, data["MLIP_name"], bold_center_align)
            col += 1

            # Normal rate  
            worksheet.write(row, col, data["Normal_rate"], number_format_2f)
            col += 1

            # Adsorbate migration rate (independent)
            worksheet.write(row, col, data["Adsorbate_migration_rate"], number_format_2f)
            col += 1

            # Anomaly sub-categories (4 columns) - excluding adsorbate migration
            worksheet.write(row, col, data["Anomaly_rate"], number_format_2f)  # total
            worksheet.write(row, col + 1, data["Reproduction_failure_rate"], number_format_2f)
            worksheet.write(row, col + 2, data["Unphysical_relaxation_rate"], number_format_2f)
            worksheet.write(row, col + 3, data["Energy_anomaly_rate"], number_format_2f)
            col += 4

            # MAE values
            worksheet.write(row, col, data["MAE_total"], number_format_3f)
            worksheet.write(row, col + 1, data["MAE_normal"], number_format_3f)
            worksheet.write(row, col + 2, data["MAE_single"], number_format_3f)
            col += 3

            # ADwT and AMDwT metrics
            worksheet.write(row, col, data["ADwT"], number_format_3f)
            worksheet.write(row, col + 1, data["AMDwT"], number_format_3f)
            col += 2

            # Numbers and time
            worksheet.write(row, col, data["Num_total"], number_format_0f)
            worksheet.write(row, col + 1, data["Time_total"], number_format_0f)

            # Time per step - fixed in seconds (no conversion)
            worksheet.write(row, col + 2, data["Time_per_step"], number_format_3f)

            worksheet.write(row, col + 3, data["Steps_total"], number_format_0f)

        # Set column widths - adjusted for independent adsorbate migration
        column_widths = [25, 18, 30, 15, 20, 25, 18, 18, 18, 18, 15, 15, 15, 18, 18, 15]
        for col, width in enumerate(column_widths):
            worksheet.set_column(col, col, width)

        # Set row heights
        for row in range(len(main_data) + 2):
            worksheet.set_row(row, 25)

    def _create_anomaly_sheet(self, writer, anomaly_data):
        """Create anomaly sheet with custom merged cell structure like main sheet."""
        workbook = writer.book
        worksheet = workbook.add_worksheet("anomaly")

        # Define formats
        header_format = workbook.add_format({
            "align": "center", 
            "valign": "vcenter",
            "text_wrap": True,
            "bold": True
        })
        center_align = workbook.add_format({"align": "center", "valign": "vcenter"})
        bold_center_align = workbook.add_format({
            "align": "center", 
            "valign": "vcenter", 
            "bold": True
        })
        number_format_0f = workbook.add_format(
            {"num_format": "#,##0", "align": "center", "valign": "vcenter"}
        )

        # Column structure:
        # 0: MLIP_name, 1: Num_normal, 2: Num_adsorbate_migration, 3-6: Anomaly count (4 columns), 7-9: Reproduction failure (3 columns), 10-13: Unphysical relaxation (4 columns)

        # Write main headers (Row 0)
        worksheet.merge_range(0, 0, 1, 0, "MLIP_name", header_format)  # 2 rows
        worksheet.merge_range(0, 1, 1, 1, "Num_normal", header_format)  # 2 rows  
        worksheet.merge_range(0, 2, 1, 2, "Num_adsorbate_migration", header_format)  # 2 rows - independent
        worksheet.merge_range(0, 3, 0, 6, "Anomaly count", header_format)  # 4 columns
        worksheet.merge_range(0, 7, 0, 9, "Reproduction failure", header_format)  # 3 columns
        worksheet.merge_range(0, 10, 0, 13, "Unphysical relaxation", header_format)  # 4 columns

        # Write anomaly sub-headers (Row 1) - excluding adsorbate migration
        anomaly_subheaders = [
            "total",
            "reproduction failure",
            "unphysical relaxation", 
            "energy anomaly"
        ]

        for col, subheader in enumerate(anomaly_subheaders, 3):  # Start from column 3
            worksheet.write(1, col, subheader, header_format)

        # Write reproduction failure sub-headers (Row 1)
        repro_subheaders = ["ads_eng_seed", "ads_seed", "slab_seed"]
        for col, subheader in enumerate(repro_subheaders, 7):
            worksheet.write(1, col, subheader, header_format)

        # Write unphysical relaxation sub-headers (Row 1)
        unphys_subheaders = ["ads_conv", "ads_move", "slab_conv", "slab_move"]
        for col, subheader in enumerate(unphys_subheaders, 10):
            worksheet.write(1, col, subheader, header_format)

        # Write data starting from row 2
        for row, data in enumerate(anomaly_data, 2):
            col = 0

            # MLIP name - bold formatting
            worksheet.write(row, col, data["MLIP_name"], bold_center_align)
            col += 1

            # Num_normal
            worksheet.write(row, col, data["Num_normal"], number_format_0f)
            col += 1

            # Num_adsorbate_migration (independent)
            worksheet.write(row, col, data["Num_adsorbate_migration"], number_format_0f)
            col += 1

            # Anomaly counts (4 columns) - excluding adsorbate migration
            worksheet.write(row, col, data["Num_anomaly_total"], number_format_0f)  # total
            worksheet.write(row, col + 1, data["Num_reproduction_failure"], number_format_0f)
            worksheet.write(row, col + 2, data["Num_unphysical_relaxation"], number_format_0f)
            worksheet.write(row, col + 3, data["Num_energy_anomaly"], number_format_0f)
            col += 4

            # Reproduction failure breakdown (3 columns)
            worksheet.write(row, col, data["ads_eng_seed"], number_format_0f)
            worksheet.write(row, col + 1, data["ads_seed"], number_format_0f)
            worksheet.write(row, col + 2, data["slab_seed"], number_format_0f)
            col += 3

            # Unphysical relaxation breakdown (4 columns)
            worksheet.write(row, col, data["ads_conv"], number_format_0f)
            worksheet.write(row, col + 1, data["ads_move"], number_format_0f)
            worksheet.write(row, col + 2, data["slab_conv"], number_format_0f)
            worksheet.write(row, col + 3, data["slab_move"], number_format_0f)

        # Set column widths - adjusted for independent adsorbate migration
        column_widths = [20, 15, 25, 15, 20, 25, 18, 15, 12, 12, 12, 12, 12, 12]
        for col, width in enumerate(column_widths):
            worksheet.set_column(col, col, width)

        # Set row heights
        for row in range(len(anomaly_data) + 2):
            worksheet.set_row(row, 25)

    def _create_mlip_sheet(self, writer, data_tmp, mlip_name):
        """Create individual MLIP sheet with custom merged cell structure like main sheet."""
        workbook = writer.book
        display_name = self._display_mlip_name(mlip_name)
        worksheet = workbook.add_worksheet(display_name)

        # Define formats
        header_format = workbook.add_format({
            "align": "center", 
            "valign": "vcenter",
            "text_wrap": True,
            "bold": True
        })
        center_align = workbook.add_format({"align": "center", "valign": "vcenter"})
        bold_center_align = workbook.add_format({
            "align": "center", 
            "valign": "vcenter", 
            "bold": True
        })
        number_format_0f = workbook.add_format(
            {"num_format": "#,##0", "align": "center", "valign": "vcenter"}
        )
        number_format_3f = workbook.add_format(
            {"num_format": "0.000", "align": "center", "valign": "vcenter"}
        )

        # Column structure:
        # 0: Adsorbate_name, 1: MAE_total, 2: MAE_normal, 3: MAE_single, 4: ADwT, 5: AMDwT, 
        # 6: Num_total, 7: Num_normal, 8: Num_adsorbate_migration, 9-12: Anomaly count (4 columns)

        # Write main headers (Row 0)
        worksheet.merge_range(0, 0, 1, 0, "Adsorbate_name", header_format)  # 2 rows
        worksheet.merge_range(0, 1, 1, 1, "MAE_total (eV)", header_format)  # 2 rows
        worksheet.merge_range(0, 2, 1, 2, "MAE_normal (eV)", header_format)  # 2 rows
        worksheet.merge_range(0, 3, 1, 3, "MAE_single (eV)", header_format)  # 2 rows
        worksheet.merge_range(0, 4, 1, 4, "ADwT (%)", header_format)  # 2 rows
        worksheet.merge_range(0, 5, 1, 5, "AMDwT (%)", header_format)  # 2 rows
        worksheet.merge_range(0, 6, 1, 6, "Num_total", header_format)  # 2 rows
        worksheet.merge_range(0, 7, 1, 7, "Num_normal", header_format)  # 2 rows
        worksheet.merge_range(0, 8, 1, 8, "Num_adsorbate_migration", header_format)  # 2 rows - independent
        worksheet.merge_range(0, 9, 0, 12, "Anomaly count", header_format)  # 4 columns

        # Write anomaly sub-headers (Row 1) - excluding adsorbate migration
        anomaly_subheaders = [
            "total",
            "reproduction failure",
            "unphysical relaxation", 
            "energy anomaly"
        ]

        for col, subheader in enumerate(anomaly_subheaders, 9):  # Start from column 9
            worksheet.write(1, col, subheader, header_format)

        # Write data starting from row 2
        for row, data in enumerate(data_tmp, 2):
            col = 0

            # Adsorbate name - bold formatting
            worksheet.write(row, col, data["Adsorbate_name"], bold_center_align)
            col += 1

            # MAE values
            worksheet.write(row, col, data["MAE_total"], number_format_3f)
            worksheet.write(row, col + 1, data["MAE_normal"], number_format_3f)
            worksheet.write(row, col + 2, data["MAE_single"], number_format_3f)
            col += 3

            # ADwT and AMDwT metrics
            worksheet.write(row, col, data["ADwT"], number_format_3f)
            worksheet.write(row, col + 1, data["AMDwT"], number_format_3f)
            col += 2

            # Numbers
            worksheet.write(row, col, data["Num_total"], number_format_0f)
            worksheet.write(row, col + 1, data["Num_normal"], number_format_0f)
            col += 2

            # Num_adsorbate_migration (independent)
            worksheet.write(row, col, data["Num_adsorbate_migration"], number_format_0f)
            col += 1

            # Anomaly counts (4 columns) - excluding adsorbate migration
            worksheet.write(row, col, data["Num_anomaly_total"], number_format_0f)  # total
            worksheet.write(row, col + 1, data["Num_reproduction_failure"], number_format_0f)
            worksheet.write(row, col + 2, data["Num_unphysical_relaxation"], number_format_0f)
            worksheet.write(row, col + 3, data["Num_energy_anomaly"], number_format_0f)

        # Set column widths - adjusted for independent adsorbate migration
        column_widths = [20, 18, 18, 18, 15, 15, 15, 15, 25, 15, 20, 25, 18]
        for col, width in enumerate(column_widths):
            worksheet.set_column(col, col, width)

        # Set row heights
        for row in range(len(data_tmp) + 2):
            worksheet.set_row(row, 25)

    def analysis(self):
        """
        Perform comprehensive MLIP benchmarking analysis.

        Automatically detects calculation structure and performs statistical analysis
        including anomaly detection, MAE calculation, and report generation.

        Note:
            If plot_enabled=False, plots will be skipped during analysis.
            All MAE calculations and Excel reports will still be generated.
        """
        # Display analysis settings
        print(f"\nAnalysis Configuration:")
        print(f"  Bond length threshold: {self.bond_length_change_threshold:.2f}")
        print(f"  Displacement threshold: {self.disp_thrs:.2f} Å")
        print(f"  Energy threshold: {self.energy_thrs:.1f} eV")
        if self.energy_cutoff is not None:
            print(f"  Energy cutoff: {self.energy_cutoff:.1f} eV")
        print()

        print("Starting analysis...")
        return self._run_analysis()

    def threshold_sensitivity_analysis(self, mode="both"):
        """
        Perform threshold sensitivity analysis to show how classification rates change with thresholds.

        Generates stacked area charts showing the proportion of each classification category
        (normal, energy_anomaly, adsorbate_migration, unphysical_relaxation, reproduction_failure)
        as threshold values vary. Uses intelligent caching to avoid recomputing identical threshold combinations.

        Args:
            mode (str): Analysis mode
                - "both": Run both displacement and bond length threshold analyses (default)
                - "disp_thrs": Vary displacement threshold from 0.05 to 2.0 Å
                - "bond_length_change_threshold": Vary bond change threshold from 5% to 50%

        Output:
            - Threshold sensitivity plots: plot/{mlip_name}/threshold_sensitivity/

        Implementation:
            Uses direct data access from result.json with vectorized operations.
        """
        # Determine which modes to run
        if mode == "both":
            modes_to_run = ["disp_thrs", "bond_length_change_threshold"]
            print(f"\nThreshold Sensitivity Analysis: Running both modes")
        elif mode in ["disp_thrs", "bond_length_change_threshold"]:
            modes_to_run = [mode]
            print(f"\nThreshold Sensitivity Analysis: Running {mode} mode")
        else:
            raise ValueError("Mode must be 'both', 'disp_thrs', or 'bond_length_change_threshold'")

        # Get MLIP list
        if self.mlip_list is None:
            self.mlip_list = sorted([
                name for name in os.listdir(self.calculating_path)
                if os.path.isdir(os.path.join(self.calculating_path, name))
                and os.path.exists(os.path.join(self.calculating_path, name, f"{name}_result.json"))
            ], key=str.lower)

        # Backup original threshold values
        original_disp_thrs = self.disp_thrs
        original_bond_threshold = self.bond_length_change_threshold

        try:
            for mlip_name in sorted(self.mlip_list, key=str.lower):
                # Load MLIP results once for all modes
                result_file = f"{self.calculating_path}/{mlip_name}/{mlip_name}_result.json"
                if not os.path.exists(result_file):
                    print(f"  Warning: Result file not found for {mlip_name}")
                    continue

                with open(result_file, "r") as f:
                    mlip_result = json.load(f)

                # Get n_crit_relax
                n_crit_relax = mlip_result.get("calculation_settings", {}).get("n_crit_relax", 999)

                # Run analysis for each mode
                for current_mode in modes_to_run:
                    print(f"Processing {mlip_name} - {current_mode} sensitivity analysis")

                    if current_mode == "disp_thrs":
                        print(f"  Mode: Varying displacement threshold (0.0 → 2.0 Å)")
                        print(f"  Fixed bond length threshold: {self.bond_length_change_threshold:.2f}")
                        threshold_values = [round(i * 0.1, 2) for i in range(0, 21)]
                        threshold_display_values = threshold_values  # Same as actual values
                        threshold_label = "Displacement Threshold (Å)"
                    else:  # bond_length_change_threshold
                        print(f"  Mode: Varying bond length threshold (0% → 50%)")
                        print(f"  Fixed displacement threshold: {self.disp_thrs:.2f} Å")
                        threshold_values = [round(i * 0.025, 3) for i in range(0, 21)]  # Actual values for calculation (0.0, 0.05, 0.1, ...)
                        threshold_display_values = [round(val * 100, 1) for val in threshold_values]  # Display as % (0, 5, 10, ...)
                        threshold_label = "Bond Length Change Threshold (%)"

                    # Use threshold analysis
                    results_data = self._threshold_analysis(
                        mlip_result, mlip_name, n_crit_relax, current_mode, 
                        threshold_values, threshold_display_values
                    )

                    # Create and save plot
                    self._create_threshold_sensitivity_plot(
                        results_data, mlip_name, current_mode, threshold_label
                    )

        finally:
            # Restore original threshold values
            self.disp_thrs = original_disp_thrs
            self.bond_length_change_threshold = original_bond_threshold

        print("Threshold sensitivity analysis completed!")

    def _threshold_analysis(self, mlip_result, mlip_name, n_crit_relax, mode, threshold_values, threshold_display_values):
        """
        Threshold analysis using vectorized operations and pre-computation.

        Key features:
        1. Pre-compute all threshold-independent values once
        2. Vectorize threshold-dependent classifications using NumPy
        3. Minimize dictionary lookups and function calls
        """
        import numpy as np

        # Step 1: Pre-process all reactions ONCE (threshold-independent)
        reactions = []
        threshold_independent_flags = []
        threshold_dependent_values = []

        has_slab_calculations = None

        for reaction_name in mlip_result:
            if reaction_name == "calculation_settings":
                continue

            reaction = mlip_result[reaction_name]

            # Apply energy cutoff filter
            if self.energy_cutoff is not None:
                try:
                    reference_energy = reaction["reference"]["ads_eng"]
                    if reference_energy > self.energy_cutoff:
                        continue
                except (KeyError, TypeError):
                    continue

            # Get calculator keys once
            calculator_keys = get_calculator_keys(reaction)
            if not calculator_keys:
                continue

            # Determine slab calculation mode once
            if has_slab_calculations is None:
                has_slab_calculations = "slab_max_disp" in reaction[calculator_keys[0]]

            # Get median calculator key
            try:
                median_calc_key = get_median_calculator_key(reaction)
            except KeyError:
                continue

            # Extract median calculator data
            median_data = reaction.get(median_calc_key, {})
            bond_change_pct = median_data.get("max_bond_change", 0.0)
            substrate_disp_median = median_data.get("substrate_displacement", 0.0)

            # Pre-compute threshold-INDEPENDENT anomalies

            # 1. Seed anomalies (highest priority - threshold independent)
            seed_anomaly = False
            if "final" in reaction:
                final_data = reaction["final"]
                seed_anomaly = (
                    (has_slab_calculations and final_data.get("slab_seed_range", 0) > self.reproduction_thrs) or
                    (has_slab_calculations and final_data.get("ads_seed_range", 0) > self.reproduction_thrs) or
                    final_data.get("ads_eng_seed_range", 0) > self.reproduction_thrs
                )

            # 2. Convergence anomalies (threshold independent)
            convergence_anomaly = False
            for calc_key in calculator_keys:
                calc_data = reaction[calc_key]
                if (has_slab_calculations and "slab_steps" in calc_data and calc_data["slab_steps"] == n_crit_relax) or \
                   ("adslab_steps" in calc_data and calc_data["adslab_steps"] == n_crit_relax):
                    convergence_anomaly = True
                    break

            # 3. Energy anomaly (threshold independent)
            energy_anomaly = False
            if "final" in reaction and "reference" in reaction:
                final_data = reaction["final"]
                if "ads_eng_median" in final_data and "ads_eng" in reaction["reference"]:
                    median_energy = final_data["ads_eng_median"]
                    reference_energy = reaction["reference"]["ads_eng"]
                    energy_diff = abs(median_energy - reference_energy)
                    energy_anomaly = energy_diff > self.energy_thrs

            # 4. Get max displacement values for threshold-dependent checks
            max_slab_disp = 0.0
            if has_slab_calculations:
                for calc_key in calculator_keys:
                    calc_data = reaction[calc_key]
                    slab_disp = calc_data.get("slab_max_disp", 0.0)
                    max_slab_disp = max(max_slab_disp, slab_disp)

            # Store pre-computed values
            reactions.append(reaction_name)
            threshold_independent_flags.append({
                'seed_anomaly': seed_anomaly,
                'convergence_anomaly': convergence_anomaly,
                'energy_anomaly': energy_anomaly
            })
            threshold_dependent_values.append({
                'bond_change_pct': bond_change_pct,
                'substrate_disp': substrate_disp_median,
                'max_slab_disp': max_slab_disp
            })

        # Convert to numpy arrays for vectorized operations
        n_reactions = len(reactions)
        if n_reactions == 0:
            return {
                "thresholds": threshold_display_values,
                "normal": [0.0] * len(threshold_values),
                "energy_anomaly": [0.0] * len(threshold_values),
                "adsorbate_migration": [0.0] * len(threshold_values),
                "unphysical_relaxation": [0.0] * len(threshold_values),
                "reproduction_failure": [0.0] * len(threshold_values)
            }

        # Create boolean arrays for threshold-independent anomalies
        seed_anomalies = np.array([f['seed_anomaly'] for f in threshold_independent_flags])
        convergence_anomalies = np.array([f['convergence_anomaly'] for f in threshold_independent_flags])
        energy_anomalies = np.array([f['energy_anomaly'] for f in threshold_independent_flags])

        # Create value arrays for threshold-dependent checks
        bond_changes = np.array([v['bond_change_pct'] for v in threshold_dependent_values])
        substrate_disps = np.array([v['substrate_disp'] for v in threshold_dependent_values])
        max_slab_disps = np.array([v['max_slab_disp'] for v in threshold_dependent_values])

        # Step 2: Vectorized classification for each threshold
        results_data = {
            "thresholds": threshold_display_values,
            "normal": [],
            "energy_anomaly": [],
            "adsorbate_migration": [],
            "unphysical_relaxation": [],
            "reproduction_failure": []
        }

        for threshold in threshold_values:
            # Vectorized threshold-dependent classifications
            if mode == "disp_thrs":
                # Displacement threshold varies
                disp_threshold = threshold
                bond_threshold_pct = self.bond_length_change_threshold * 100
                displacement_anomalies = convergence_anomalies | (max_slab_disps > disp_threshold) | (substrate_disps > disp_threshold)
            else:  # bond_length_change_threshold mode
                # Bond threshold varies
                disp_threshold = self.disp_thrs
                bond_threshold_pct = threshold * 100
                displacement_anomalies = convergence_anomalies | (max_slab_disps > disp_threshold) | (substrate_disps > disp_threshold)

            migration_anomalies = bond_changes > bond_threshold_pct

            # Hierarchical classification using vectorized operations
            classifications = np.zeros(n_reactions, dtype=int)
            # 0 = normal, 1 = energy_anomaly, 2 = adsorbate_migration, 3 = unphysical_relaxation, 4 = reproduction_failure

            # Apply hierarchical rules (order matters!)
            classifications[seed_anomalies] = 4  # Highest priority
            classifications[(classifications == 0) & displacement_anomalies] = 3
            classifications[(classifications == 0) & migration_anomalies] = 2
            classifications[(classifications == 0) & energy_anomalies] = 1
            # Remaining zeros are normal

            # Count each category
            unique, counts = np.unique(classifications, return_counts=True)
            category_counts = dict(zip(unique, counts))

            # Calculate percentages
            total = n_reactions
            results_data["normal"].append(category_counts.get(0, 0) / total * 100)
            results_data["energy_anomaly"].append(category_counts.get(1, 0) / total * 100)
            results_data["adsorbate_migration"].append(category_counts.get(2, 0) / total * 100)
            results_data["unphysical_relaxation"].append(category_counts.get(3, 0) / total * 100)
            results_data["reproduction_failure"].append(category_counts.get(4, 0) / total * 100)

        return results_data

    def _create_threshold_sensitivity_plot(self, results_data, mlip_name, mode, threshold_label):
        """Create and save threshold sensitivity plot."""
        # Setup plot directory
        display_name = self._display_mlip_name(mlip_name)
        plot_path = os.path.join(os.getcwd(), "plot", display_name, "threshold_sensitivity")
        os.makedirs(plot_path, exist_ok=True)

        # Setup font if specified
        if self.font_setting:
            from catbench.utils.analysis_utils import set_matplotlib_font
            set_matplotlib_font(self.font_setting[0], self.font_setting[1])

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Define colors for each category (professional color scheme)
        colors = {
            "reproduction_failure": "#2D234C",   # Dark purple (most severe)
            "unphysical_relaxation": "#782E7C",  # Light purple (physical issue)
            "adsorbate_migration": "#C14B65",    # Pink/coral (migration issue)
            "energy_anomaly": "#F48430",         # Orange (energy issue)
            "normal": "#F6DB57"                  # Yellow (stable)
        }
        # Create stacked area plot
        thresholds = results_data["thresholds"]

        # Prepare data for stacking (bottom to top)
        categories = ["normal", "energy_anomaly", "adsorbate_migration", "unphysical_relaxation", "reproduction_failure"]
        category_labels = ["Normal", "Energy Anomaly", "Adsorbate Migration", "Unphysical Relaxation", "Reproduction Failure"]

        # Create cumulative data for stacking
        cumulative = np.zeros(len(thresholds))

        for i, (category, label) in enumerate(zip(categories, category_labels)):
            values = np.array(results_data[category])

            # Create filled area
            ax.fill_between(
                thresholds, 
                cumulative, 
                cumulative + values,
                color=colors[category],
                alpha=1.0,
                label=label,
                linewidth=0.8,
                edgecolor='black'
            )

            cumulative += values

        # Customize plot
        if not self.xlabel_off:
            ax.set_xlabel(threshold_label, fontsize=25)
        if not self.ylabel_off:
            ax.set_ylabel("Rate (%)", fontsize=25)

        # Set axis limits based on threshold range
        threshold_min = min(thresholds)
        threshold_max = max(thresholds)

        ax.set_xlim(threshold_min, threshold_max)
        ax.set_ylim(0, 100)
        # Tick bins control for threshold plot
        if self.x_tick_bins is not None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=self.x_tick_bins))
        if self.y_tick_bins is not None:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=self.y_tick_bins))
        # Tick label formatting for threshold plot
        if self.tick_decimal_places is not None:
            fmt = f"%.{self.tick_decimal_places}f"
            ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
            ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

        # Grid
        if self.grid:
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
        else:
            ax.grid(False)

        # Legend (reversed order for top-to-bottom display)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1], labels[::-1],  # Reverse the order
            loc='lower right',
            fontsize=self.legend_fontsize,
            frameon=True,
            fancybox=True,
            shadow=True,
            markerscale=self.legend_markerscale,
        )

        # Tick formatting
        ax.tick_params(axis='both', which='major', labelsize=self.tick_labelsize)

        # Set spine properties
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # Tight layout
        plt.tight_layout()

        # Save plot
        if mode == "disp_thrs":
            filename = f"{display_name}_disp_thrs_sensitivity.png"
        else:
            filename = f"{display_name}_bond_threshold_sensitivity.png"

        plt.savefig(
            os.path.join(plot_path, filename), 
            dpi=self.dpi, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close()

        print(f"  ✅ Saved: {filename}")

    def _run_analysis(self):
        """Execute comprehensive analysis with anomaly detection and report generation."""
        print("Processing MLIP results...")
        if not self.plot_enabled:
            print("Note: Plot generation disabled")

        main_data = []
        anomaly_data = []
        MLIP_datas = {}

        # Get MLIP list
        if self.mlip_list is None:
            self.mlip_list = sorted([
                name for name in os.listdir(self.calculating_path)
                if os.path.isdir(os.path.join(self.calculating_path, name))
                and os.path.exists(os.path.join(self.calculating_path, name, f"{name}_result.json"))
            ], key=str.lower)

        for mlip_name in sorted(self.mlip_list, key=str.lower):
            adsorbates = set()
            print(f"Processing {self._display_mlip_name(mlip_name)}")

            # Load results first
            with open(f"{self.calculating_path}/{mlip_name}/{mlip_name}_result.json", "r") as f:
                mlip_result = json.load(f)

            # Get n_crit_relax from calculation settings (with fallback to default)
            n_crit_relax = mlip_result.get("calculation_settings", {}).get("n_crit_relax", 999)

            # Perform anomaly detection with configured thresholds
            MLIP_anomaly, anomaly_summary = self._anomaly_detection(mlip_result, mlip_name, n_crit_relax)

            # Save anomaly detection results
            os.makedirs(f"{self.calculating_path}/{mlip_name}", exist_ok=True)
            anomaly_detection_save = {
                "thresholds": MLIP_anomaly["thresholds"],
                "normal": MLIP_anomaly["normal"],
                "energy_anomaly": MLIP_anomaly["energy_anomaly"],
                "adsorbate_migration": MLIP_anomaly["adsorbate_migration"],
                "unphysical_relaxation": MLIP_anomaly["unphysical_relaxation"],
                "reproduction_failure": MLIP_anomaly["reproduction_failure"]
            }
            save_anomaly_detection_results(
                self.calculating_path, mlip_name, anomaly_detection_save
            )

            # Initialize data structures for 5-category classification
            ads_data = {
                "all": {
                    "normal": {"DFT": [], "MLIP": [], 
                             "MLIP_min": [], "MLIP_max": []},
                    "energy_anomaly": {"DFT": [], "MLIP": [], 
                                     "MLIP_min": [], "MLIP_max": []},
                    "adsorbate_migration": {"DFT": [], "MLIP": [], 
                                          "MLIP_min": [], "MLIP_max": []},
                    "unphysical_relaxation": {"DFT": [], "MLIP": [], 
                                            "MLIP_min": [], "MLIP_max": []},
                    "reproduction_failure": {"DFT": [], "MLIP": [], 
                                           "MLIP_min": [], "MLIP_max": []},
                }
            }

            # Check if absolute energy MLIP (has slab calculations)
            first_reaction = True
            absolute_energy_MLIP = True
            for reaction in mlip_result:
                if reaction == "calculation_settings":
                    continue

                adsorbate = find_adsorbate(mlip_result[reaction]["reference"])
                if adsorbate:
                    adsorbates.add(adsorbate)

                if first_reaction:
                    first_reaction = False
                    if "slab_conv" not in anomaly_summary[reaction]["details"]:
                        absolute_energy_MLIP = False
            # Determine analysis adsorbates
            analysis_adsorbates = self._get_analysis_adsorbates(adsorbates)

            # Initialize counters
            time_accum = 0
            step_accum = 0
            slab_conv = ads_conv = slab_move = ads_move = 0
            slab_seed = ads_seed = ads_eng_seed = 0

            # Process each reaction
            for reaction in mlip_result:
                if reaction == "calculation_settings":
                    continue

                adsorbate = find_adsorbate(mlip_result[reaction]["reference"])

                # Apply energy cutoff filter (only if cutoff is specified)
                if self.energy_cutoff is not None:
                    try:
                        reference_energy = mlip_result[reaction]["reference"]["ads_eng"]
                        if reference_energy > self.energy_cutoff:
                            continue  # Skip reactions with energy above cutoff
                    except (KeyError, TypeError):
                        continue  # Skip if reference energy is not available

                if adsorbate and adsorbate in analysis_adsorbates:
                    if adsorbate not in ads_data:
                        ads_data[adsorbate] = {
                            "normal": {"DFT": [], "MLIP": [], 
                                     "MLIP_min": [], "MLIP_max": []},
                            "energy_anomaly": {"DFT": [], "MLIP": [], 
                                             "MLIP_min": [], "MLIP_max": []},
                            "adsorbate_migration": {"DFT": [], "MLIP": [], 
                                                  "MLIP_min": [], "MLIP_max": []},
                            "unphysical_relaxation": {"DFT": [], "MLIP": [], 
                                                    "MLIP_min": [], "MLIP_max": []},
                            "reproduction_failure": {"DFT": [], "MLIP": [], 
                                                   "MLIP_min": [], "MLIP_max": []},
                        }

                    classification = anomaly_summary[reaction]["classification"]
                    MLIP_min, MLIP_max = get_ads_eng_range(mlip_result[reaction])

                    try:
                        # Add data to appropriate classification category
                        dft_value = mlip_result[reaction]["reference"]["ads_eng"]
                        mlip_value = mlip_result[reaction]["final"]["ads_eng_median"]

                        # Add to adsorbate-specific data
                        ads_data[adsorbate][classification]["DFT"].append(dft_value)
                        ads_data[adsorbate][classification]["MLIP"].append(mlip_value)
                        ads_data[adsorbate][classification]["MLIP_min"].append(MLIP_min)
                        ads_data[adsorbate][classification]["MLIP_max"].append(MLIP_max)

                        # Add to overall data
                        ads_data["all"][classification]["DFT"].append(dft_value)
                        ads_data["all"][classification]["MLIP"].append(mlip_value)
                        ads_data["all"][classification]["MLIP_min"].append(MLIP_min)
                        ads_data["all"][classification]["MLIP_max"].append(MLIP_max)
                    except Exception as e:
                        print(f"Error processing reaction {reaction}: {str(e)}")
                        continue

                    # Accumulate time and steps (directly from saved results)
                    time_accum += sum(
                        value for key, value in mlip_result[reaction]["final"].items()
                        if "time_total" in key
                    )

                    step_accum += sum(
                        value for key, value in mlip_result[reaction]["final"].items()
                        if "steps_total" in key
                    )

                    # Count anomalies (using current anomaly detection results)
                    reaction_anomalies = anomaly_summary[reaction]["details"]

                    if absolute_energy_MLIP:
                        if reaction_anomalies["slab_conv"]:
                            slab_conv += 1

                    if reaction_anomalies["ads_conv"]:
                        ads_conv += 1                    

                    if absolute_energy_MLIP:
                        if reaction_anomalies["slab_move"]:
                            slab_move += 1

                    if reaction_anomalies["ads_move"]:
                            ads_move += 1

                    if absolute_energy_MLIP:
                        if reaction_anomalies["slab_seed"]:
                            slab_seed += 1

                    if absolute_energy_MLIP:
                        if reaction_anomalies["ads_seed"]:
                            ads_seed += 1

                    if reaction_anomalies["ads_eng_seed"]:
                        ads_eng_seed += 1
                        
            # Convert lists to numpy arrays for all categories
            for category in ads_data:
                for classification in ads_data[category]:
                    for key in ads_data[category][classification]:
                        ads_data[category][classification][key] = np.array(ads_data[category][classification][key])

            # Cache the mlip_result for later use in Excel generation
            self._mlip_result_cache[mlip_name] = mlip_result

            # Generate plots and calculate MAEs
            all_dft_arrays = [ads_data["all"]["normal"]["DFT"]]
            for anomaly_type in ["energy_anomaly", "adsorbate_migration", "unphysical_relaxation", "reproduction_failure"]:
                all_dft_arrays.append(ads_data["all"][anomaly_type]["DFT"])

            # Filter out empty arrays before concatenation
            non_empty_arrays = [arr for arr in all_dft_arrays if len(arr) > 0]
            DFT_data = np.concatenate(non_empty_arrays) if non_empty_arrays else np.array([])
            if len(DFT_data) > 0:
                if self.min_value is None or self.max_value is None:
                    min_value, max_value = min_max(DFT_data)
                    if self.min_value is not None:
                        min_value = self.min_value
                    if self.max_value is not None:
                        max_value = self.max_value
                else:
                    min_value, max_value = self.min_value, self.max_value

                # Re-sort adsorbates by data count for consistent ordering across all plots
                analysis_adsorbates = sorted(analysis_adsorbates, key=lambda ads: self._get_adsorbate_count(ads, ads_data), reverse=True)

                # Create single calculation plots data with re-sorted adsorbates
                single_data = self._create_single_data_structure(mlip_result, analysis_adsorbates)

                if self.plot_enabled:
                    print(f"  Generating plots for {self._display_mlip_name(mlip_name)}...")
                    # Setup plot directories once
                    plot_save_path, mono_path, multi_path = self._setup_plot(mlip_name)

                    # Generate all plots
                    MAE_total, MAE_normal, MAEs_total_multi, MAEs_normal_multi = self._plot_generator(
                        ads_data, single_data, mlip_name, min_value, max_value, mono_path, multi_path
                    )
                else:
                    print(f"  Skipping plot generation for {self._display_mlip_name(mlip_name)} (plot_enabled=False)")
                    # Calculate MAEs without generating plots
                    MAE_total = self._calculate_mae_from_data(ads_data, ["normal", "adsorbate_migration", "energy_anomaly", "unphysical_relaxation", "reproduction_failure"])
                    MAE_normal = self._calculate_mae_from_data(ads_data, ["normal"])

                    MAEs_total_multi = self._calculate_maes_by_adsorbate(ads_data, ["normal", "adsorbate_migration", "energy_anomaly", "unphysical_relaxation", "reproduction_failure"], analysis_adsorbates)
                    MAEs_normal_multi = self._calculate_maes_by_adsorbate(ads_data, ["normal"], analysis_adsorbates)

                # Calculate single MAE (using single_calculation vs reference)
                single_mae = self._calculate_single_mae(mlip_result, analysis_adsorbates)

                # Calculate single MAE by adsorbate for MLIP sheet
                single_mae_by_adsorbate = self._calculate_single_mae_by_adsorbate(mlip_result, analysis_adsorbates)

                # Calculate ADwT and AMDwT metrics
                print(f"  🔧 Calculating ADwT metric for {self._display_mlip_name(mlip_name)}...")
                adwt_value = self._calculate_adwt(mlip_result, analysis_adsorbates)
                # Cache ADwT calculations for each adsorbate
                for adsorbate in analysis_adsorbates:
                    cache_key = (mlip_name, adsorbate)
                    if cache_key not in self._adwt_cache:
                        self._adwt_cache[cache_key] = self._calculate_adwt_by_adsorbate(mlip_result, adsorbate)

                print(f"  🔧 Calculating AMDwT metric for {self._display_mlip_name(mlip_name)}...")
                amdwt_value = self._calculate_amdwt(mlip_result, analysis_adsorbates)
                # Cache AMDwT calculations for each adsorbate
                for adsorbate in analysis_adsorbates:
                    cache_key = (mlip_name, adsorbate)
                    if cache_key not in self._amdwt_cache:
                        self._amdwt_cache[cache_key] = self._calculate_amdwt_by_adsorbate(mlip_result, adsorbate)

                MLIP_datas[mlip_name] = {
                    "total": MAEs_total_multi,
                    "normal": MAEs_normal_multi,
                    "single_mae": single_mae_by_adsorbate,
                    "ads_data": ads_data,  # Add full ads_data for anomaly breakdown
                }

                # Prepare main data  
                normal_count = len(ads_data["all"]["normal"]["DFT"])
                # Anomaly count excludes adsorbate migration (now independent category)
                anomaly_count = (len(ads_data["all"]["energy_anomaly"]["DFT"]) + 
                               len(ads_data["all"]["unphysical_relaxation"]["DFT"]) + 
                               len(ads_data["all"]["reproduction_failure"]["DFT"]))
                total_num = normal_count + anomaly_count + len(ads_data["all"]["adsorbate_migration"]["DFT"])
                anomaly_rate = (anomaly_count / total_num) * 100 if total_num > 0 else 0

                # Calculate individual anomaly rates
                reproduction_failure_rate = (len(ads_data["all"]["reproduction_failure"]["DFT"]) / total_num) * 100 if total_num > 0 else 0
                unphysical_relaxation_rate = (len(ads_data["all"]["unphysical_relaxation"]["DFT"]) / total_num) * 100 if total_num > 0 else 0
                adsorbate_migration_rate = (len(ads_data["all"]["adsorbate_migration"]["DFT"]) / total_num) * 100 if total_num > 0 else 0
                energy_anomaly_rate = (len(ads_data["all"]["energy_anomaly"]["DFT"]) / total_num) * 100 if total_num > 0 else 0
                normal_rate = (normal_count / total_num) * 100 if total_num > 0 else 0

                main_data.append({
                    "MLIP_name": self._display_mlip_name(mlip_name),
                    "Normal_rate": normal_rate,
                    "Anomaly_rate": anomaly_rate,
                    "Reproduction_failure_rate": reproduction_failure_rate,
                    "Unphysical_relaxation_rate": unphysical_relaxation_rate,
                    "Adsorbate_migration_rate": adsorbate_migration_rate,
                    "Energy_anomaly_rate": energy_anomaly_rate,
                    "MAE_total": MAE_total,
                    "MAE_normal": MAE_normal,
                    "MAE_single": single_mae,
                    "ADwT": adwt_value,
                    "AMDwT": amdwt_value,
                    "Num_total": total_num,
                    "Time_total": time_accum,
                    "Time_per_step": time_accum / step_accum if step_accum > 0 else 0,
                    "Steps_total": step_accum,
                })

                # Prepare anomaly data with specified order
                anomaly_data_dict = {
                    "MLIP_name": self._display_mlip_name(mlip_name),
                    "Num_normal": normal_count,
                    "Num_anomaly_total": anomaly_count,
                    "Num_reproduction_failure": len(ads_data["all"]["reproduction_failure"]["DFT"]),
                    "Num_unphysical_relaxation": len(ads_data["all"]["unphysical_relaxation"]["DFT"]),
                    "Num_adsorbate_migration": len(ads_data["all"]["adsorbate_migration"]["DFT"]),
                    "Num_energy_anomaly": len(ads_data["all"]["energy_anomaly"]["DFT"]),
                    # Reproduction failure breakdown
                    "ads_eng_seed": ads_eng_seed,
                    "ads_seed": ads_seed if absolute_energy_MLIP else 0,
                    "slab_seed": slab_seed if absolute_energy_MLIP else 0,
                    # Unphysical relaxation breakdown
                    "ads_conv": ads_conv,
                    "ads_move": ads_move,
                    "slab_conv": slab_conv if absolute_energy_MLIP else 0,
                    "slab_move": slab_move if absolute_energy_MLIP else 0,
                }

                anomaly_data.append(anomaly_data_dict)

        # Create Excel output
        self._create_excel_output(main_data, anomaly_data, MLIP_datas, list(analysis_adsorbates))


    def _create_single_data_structure(self, mlip_result, analysis_adsorbates):
        """Create data structure for single calculation plotting."""
        single_data = {
            "all": {
                "all": {"DFT": [], "MLIP": []}
            }
        }

        # Initialize adsorbate-specific data
        for adsorbate in analysis_adsorbates:
            single_data[adsorbate] = {
                "all": {"DFT": [], "MLIP": []}
            }

        # Collect single calculation data
        for reaction in mlip_result:
            if reaction == "calculation_settings":
                continue

            adsorbate = find_adsorbate(mlip_result[reaction]["reference"])

            # Apply energy cutoff filter (only if cutoff is specified)
            if self.energy_cutoff is not None:
                try:
                    reference_energy = mlip_result[reaction]["reference"]["ads_eng"]
                    if reference_energy > self.energy_cutoff:
                        continue  # Skip reactions with energy above cutoff
                except (KeyError, TypeError):
                    continue  # Skip if reference energy is not available

            if adsorbate and adsorbate in analysis_adsorbates:
                dft_value = mlip_result[reaction]["reference"]["ads_eng"]
                mlip_value = mlip_result[reaction]["single_calculation"]["ads_eng"]

                # Add to all data
                single_data["all"]["all"]["DFT"].append(dft_value)
                single_data["all"]["all"]["MLIP"].append(mlip_value)

                # Add to adsorbate-specific data
                single_data[adsorbate]["all"]["DFT"].append(dft_value)
                single_data[adsorbate]["all"]["MLIP"].append(mlip_value)

        # Convert lists to numpy arrays
        for category in single_data:
            for subcategory in single_data[category]:
                for key in single_data[category][subcategory]:
                    single_data[category][subcategory][key] = np.array(single_data[category][subcategory][key])

        return single_data

    def _calculate_mae_from_data(self, ads_data, types):
        """Calculate MAE from data without generating plots."""
        plot_data = prepare_plot_data(ads_data, types)
        if len(plot_data["DFT"]) > 0:
            return np.sum(np.abs(plot_data["DFT"] - plot_data["MLIP"])) / len(plot_data["DFT"])
        else:
            return 0.0

    def _calculate_maes_by_adsorbate(self, ads_data, types, analysis_adsorbates):
        """Calculate MAEs by adsorbate without generating plots."""
        MAEs = {}
        error_sum = 0
        len_total = 0

        for adsorbate in analysis_adsorbates:
            plot_data = prepare_plot_data(ads_data, types, adsorbate)

            mae = (
                np.sum(np.abs(plot_data["DFT"] - plot_data["MLIP"])) / len(plot_data["DFT"])
                if len(plot_data["DFT"]) != 0 else 0
            )
            MAEs[adsorbate] = mae
            error_sum += np.sum(np.abs(plot_data["DFT"] - plot_data["MLIP"]))
            len_total += len(plot_data["DFT"])
            MAEs[f"len_{adsorbate}"] = len(plot_data["DFT"])

        # Calculate total MAE
        MAE_total = error_sum / len_total if len_total != 0 else 0
        MAEs["total"] = MAE_total

        return MAEs

    def _calculate_single_mae_by_adsorbate(self, mlip_result, analysis_adsorbates):
        """Calculate MAE for single-point calculations vs reference by adsorbate."""
        adsorbate_mae = {}

        for target_adsorbate in analysis_adsorbates:
            dft_values = []
            mlip_values = []

            for reaction in mlip_result:
                if reaction == "calculation_settings":
                    continue

                adsorbate = find_adsorbate(mlip_result[reaction]["reference"])

                # Apply energy cutoff filter (only if cutoff is specified)
                if self.energy_cutoff is not None:
                    try:
                        reference_energy = mlip_result[reaction]["reference"]["ads_eng"]
                        if reference_energy > self.energy_cutoff:
                            continue  # Skip reactions with energy above cutoff
                    except (KeyError, TypeError):
                        continue  # Skip if reference energy is not available

                if adsorbate and adsorbate == target_adsorbate:
                    dft_values.append(mlip_result[reaction]["reference"]["ads_eng"])
                    mlip_values.append(mlip_result[reaction]["single_calculation"]["ads_eng"])

            if len(dft_values) > 0:
                dft_array = np.array(dft_values)
                mlip_array = np.array(mlip_values)
                adsorbate_mae[target_adsorbate] = np.mean(np.abs(dft_array - mlip_array))
            else:
                adsorbate_mae[target_adsorbate] = 0.0

        return adsorbate_mae

    def _calculate_single_mae(self, mlip_result, analysis_adsorbates):
        """Calculate MAE for single-point calculations vs reference."""
        dft_values = []
        mlip_values = []

        for reaction in mlip_result:
            if reaction == "calculation_settings":
                continue

            # Filter by adsorbate type (same logic as main analysis loop)
            adsorbate = find_adsorbate(mlip_result[reaction]["reference"])

            # Apply energy cutoff filter (only if cutoff is specified)
            if self.energy_cutoff is not None:
                try:
                    reference_energy = mlip_result[reaction]["reference"]["ads_eng"]
                    if reference_energy > self.energy_cutoff:
                        continue  # Skip reactions with energy above cutoff
                except (KeyError, TypeError):
                    continue  # Skip if reference energy is not available

            if adsorbate and adsorbate in analysis_adsorbates:
                dft_values.append(mlip_result[reaction]["reference"]["ads_eng"])
                mlip_values.append(mlip_result[reaction]["single_calculation"]["ads_eng"])

        if len(dft_values) > 0:
            dft_array = np.array(dft_values)
            mlip_array = np.array(mlip_values)
            return np.mean(np.abs(dft_array - mlip_array))
        else:
            return 0.0

    def _calculate_adwt(self, mlip_result, analysis_adsorbates):
        """
        Calculate ADwT (Average Distance within Threshold) metric.

        ADwT is the average DwT across thresholds ranging from 0.01 to 0.5 Å in increments of 0.001 Å.
        DwT is computed as the percentage of structures with position MAE below the threshold.

        Uses vectorized operations for efficient calculation.
        """

        # Pre-collect all position MAE values once
        pos_mae_values = []

        for reaction in mlip_result:
            if reaction == "calculation_settings":
                continue

            # Filter by adsorbate and energy cutoff (same as main analysis)
            adsorbate = find_adsorbate(mlip_result[reaction]["reference"])

            if self.energy_cutoff is not None:
                try:
                    reference_energy = mlip_result[reaction]["reference"]["ads_eng"]
                    if reference_energy > self.energy_cutoff:
                        continue
                except (KeyError, TypeError):
                    continue

            if adsorbate and adsorbate in analysis_adsorbates:
                # Check all calculators
                calculator_keys = [key for key in mlip_result[reaction] 
                                 if isinstance(key, (int, str)) and str(key).isdigit()]

                for calc_key in calculator_keys:
                    calc_data = mlip_result[reaction][calc_key]

                    # Collect slab position MAE if available
                    if "slab_pos_mae" in calc_data:
                        pos_mae_values.append(calc_data["slab_pos_mae"])

                    # Collect adslab position MAE
                    if "adslab_pos_mae" in calc_data:
                        pos_mae_values.append(calc_data["adslab_pos_mae"])

        if not pos_mae_values:
            return 0.0

        # Convert to numpy array for fast operations
        pos_mae_array = np.array(pos_mae_values)
        total_structures = len(pos_mae_array)

        # Generate thresholds and calculate DwT values (original loop-based approach)
        thresholds = [round(0.01 + i * 0.001, 3) for i in range(490)]  # 0.01, 0.011, ..., 0.5

        # Original loop-based calculation for exact compatibility
        dwt_values = []
        for threshold in thresholds:
            within_threshold_count = np.sum(pos_mae_array < threshold)
            dwt = (within_threshold_count / total_structures) * 100
            dwt_values.append(dwt)

        # Calculate ADwT as average of all DwT values
        adwt_result = sum(dwt_values) / len(dwt_values)
        print(f"  ADwT: {adwt_result:.1f}%")
        return adwt_result

    def _calculate_amdwt(self, mlip_result, analysis_adsorbates):
        """
        Calculate AMDwT (Average Max Displacement within Threshold) metric.

        AMDwT is the average MDwT across thresholds ranging from 0.01 to 2.0 Å in increments of 0.001 Å.
        MDwT is computed as the percentage of structures with max displacement below the threshold.

        Uses vectorized operations for efficient calculation.
        """

        # Pre-collect all max displacement values once
        max_disp_values = []

        for reaction in mlip_result:
            if reaction == "calculation_settings":
                continue

            # Filter by adsorbate and energy cutoff (same as main analysis)
            adsorbate = find_adsorbate(mlip_result[reaction]["reference"])

            if self.energy_cutoff is not None:
                try:
                    reference_energy = mlip_result[reaction]["reference"]["ads_eng"]
                    if reference_energy > self.energy_cutoff:
                        continue
                except (KeyError, TypeError):
                    continue

            if adsorbate and adsorbate in analysis_adsorbates:
                # Check all calculators
                calculator_keys = [key for key in mlip_result[reaction] 
                                 if isinstance(key, (int, str)) and str(key).isdigit()]

                for calc_key in calculator_keys:
                    calc_data = mlip_result[reaction][calc_key]

                    # Collect slab max displacement if available
                    if "slab_max_disp" in calc_data:
                        max_disp_values.append(calc_data["slab_max_disp"])

                    # Collect adslab max displacement
                    if "adslab_max_disp" in calc_data:
                        max_disp_values.append(calc_data["adslab_max_disp"])

        if not max_disp_values:
            return 0.0

        # Convert to numpy array for fast operations
        max_disp_array = np.array(max_disp_values)
        total_structures = len(max_disp_array)

        # Generate thresholds and calculate MDwT values (original loop-based approach)
        thresholds = [round(0.01 + i * 0.001, 3) for i in range(1490)]  # 0.01, 0.011, ..., 1.5

        # Original loop-based calculation for exact compatibility
        mdwt_values = []
        for threshold in thresholds:
            within_threshold_count = np.sum(max_disp_array < threshold)
            mdwt = (within_threshold_count / total_structures) * 100
            mdwt_values.append(mdwt)

        # Calculate AMDwT as average of all MDwT values
        amdwt_result = sum(mdwt_values) / len(mdwt_values)
        print(f"  AMDwT: {amdwt_result:.1f}%")
        return amdwt_result

    def _calculate_adwt_by_adsorbate(self, mlip_result, target_adsorbate):
        """
        Calculate ADwT metric for a specific adsorbate.

        Uses vectorized operations for efficient calculation.
        """
        cache_key = (id(mlip_result), target_adsorbate, 'adwt')
        if hasattr(self, '_calculation_cache') and cache_key in self._calculation_cache:
            return self._calculation_cache[cache_key]
        
        # Pre-collect all position MAE values for this adsorbate once
        pos_mae_values = []

        for reaction in mlip_result:
            if reaction == "calculation_settings":
                continue

            # Filter by adsorbate and energy cutoff
            adsorbate = find_adsorbate(mlip_result[reaction]["reference"])

            if self.energy_cutoff is not None:
                try:
                    reference_energy = mlip_result[reaction]["reference"]["ads_eng"]
                    if reference_energy > self.energy_cutoff:
                        continue
                except (KeyError, TypeError):
                    continue

            if adsorbate and adsorbate == target_adsorbate:
                # Check all calculators
                calculator_keys = [key for key in mlip_result[reaction] 
                                 if isinstance(key, (int, str)) and str(key).isdigit()]

                for calc_key in calculator_keys:
                    calc_data = mlip_result[reaction][calc_key]

                    # Collect slab position MAE if available
                    if "slab_pos_mae" in calc_data:
                        pos_mae_values.append(calc_data["slab_pos_mae"])

                    # Collect adslab position MAE
                    if "adslab_pos_mae" in calc_data:
                        pos_mae_values.append(calc_data["adslab_pos_mae"])

        if not pos_mae_values:
            return 0.0

        # Convert to numpy array for fast operations
        pos_mae_array = np.array(pos_mae_values)
        total_structures = len(pos_mae_array)

        # Generate thresholds and calculate DwT values (original loop-based approach)
        thresholds = [round(0.01 + i * 0.001, 3) for i in range(490)]  # 0.01, 0.011, ..., 0.5

        # Original loop-based calculation for exact compatibility
        dwt_values = []
        for threshold in thresholds:
            within_threshold_count = np.sum(pos_mae_array < threshold)
            dwt = (within_threshold_count / total_structures) * 100
            dwt_values.append(dwt)

        # Calculate ADwT as average of all DwT values
        result = sum(dwt_values) / len(dwt_values)
        
        # Store in cache
        if hasattr(self, '_calculation_cache'):
            cache_key = (id(mlip_result), target_adsorbate, 'adwt')
            self._calculation_cache[cache_key] = result
        
        return result

    def _calculate_amdwt_by_adsorbate(self, mlip_result, target_adsorbate):
        """
        Calculate AMDwT metric for a specific adsorbate.

        Uses vectorized operations for efficient calculation.
        """
        cache_key = (id(mlip_result), target_adsorbate, 'amdwt')
        if hasattr(self, '_calculation_cache') and cache_key in self._calculation_cache:
            return self._calculation_cache[cache_key]
        
        # Pre-collect all max displacement values for this adsorbate once
        max_disp_values = []

        for reaction in mlip_result:
            if reaction == "calculation_settings":
                continue

            # Filter by adsorbate and energy cutoff
            adsorbate = find_adsorbate(mlip_result[reaction]["reference"])

            if self.energy_cutoff is not None:
                try:
                    reference_energy = mlip_result[reaction]["reference"]["ads_eng"]
                    if reference_energy > self.energy_cutoff:
                        continue
                except (KeyError, TypeError):
                    continue

            if adsorbate and adsorbate == target_adsorbate:
                # Check all calculators
                calculator_keys = [key for key in mlip_result[reaction] 
                                 if isinstance(key, (int, str)) and str(key).isdigit()]

                for calc_key in calculator_keys:
                    calc_data = mlip_result[reaction][calc_key]

                    # Collect slab max displacement if available
                    if "slab_max_disp" in calc_data:
                        max_disp_values.append(calc_data["slab_max_disp"])

                    # Collect adslab max displacement
                    if "adslab_max_disp" in calc_data:
                        max_disp_values.append(calc_data["adslab_max_disp"])

        if not max_disp_values:
            return 0.0

        # Convert to numpy array for fast operations
        max_disp_array = np.array(max_disp_values)
        total_structures = len(max_disp_array)

        # Generate thresholds and calculate MDwT values (original loop-based approach)
        thresholds = [round(0.01 + i * 0.001, 3) for i in range(1490)]  # 0.01, 0.011, ..., 1.5

        # Original loop-based calculation for exact compatibility
        mdwt_values = []
        for threshold in thresholds:
            within_threshold_count = np.sum(max_disp_array < threshold)
            mdwt = (within_threshold_count / total_structures) * 100
            mdwt_values.append(mdwt)

        # Calculate AMDwT as average of all MDwT values
        result = sum(mdwt_values) / len(mdwt_values)
        
        # Store in cache
        if hasattr(self, '_calculation_cache'):
            cache_key = (id(mlip_result), target_adsorbate, 'amdwt')
            self._calculation_cache[cache_key] = result
        
        return result

    def _plot_generator(self, ads_data, single_data, mlip_name, min_value, max_value, mono_path, multi_path):
        """
        Generate all plots efficiently.

        This function:
        1. Pre-computes all plot data once
        2. Generates mono and multi plots systematically

        Returns:
            tuple: (MAE_total, MAE_normal, MAEs_total_multi, MAEs_normal_multi)
        """
        # Pre-process all plot data once
        plot_configs = {
            "total": ["normal", "adsorbate_migration", "energy_anomaly", "unphysical_relaxation", "reproduction_failure"],
            "normal": ["normal"], 
            "migration": ["adsorbate_migration"],
            "anomaly": ["energy_anomaly", "unphysical_relaxation", "reproduction_failure"]
        }

        # Pre-compute all plot data
        plot_data_cache = {}
        for tag, types in plot_configs.items():
            plot_data_cache[tag] = prepare_plot_data(ads_data, types)

        # Handle single data
        if len(single_data["all"]["all"]["DFT"]) > 0:
            plot_data_cache["single"] = prepare_plot_data(single_data, ["all"])

        # Generate mono plots with pre-computed data
        MAE_total = self.mono_plotter(ads_data, mlip_name, "total", min_value, max_value, mono_path, plot_data_cache.get("total"))
        MAE_normal = self.mono_plotter(ads_data, mlip_name, "normal", min_value, max_value, mono_path, plot_data_cache.get("normal"))
        self.mono_plotter(ads_data, mlip_name, "migration", min_value, max_value, mono_path, plot_data_cache.get("migration"))
        self.mono_plotter(ads_data, mlip_name, "anomaly", min_value, max_value, mono_path, plot_data_cache.get("anomaly"))

        # Generate multi plots
        MAEs_total_multi = self.multi_plotter(ads_data, mlip_name, plot_configs["total"], "total", min_value, max_value, multi_path)
        MAEs_normal_multi = self.multi_plotter(ads_data, mlip_name, plot_configs["normal"], "normal", min_value, max_value, multi_path)
        self.multi_plotter(ads_data, mlip_name, plot_configs["migration"], "migration", min_value, max_value, multi_path)
        self.multi_plotter(ads_data, mlip_name, plot_configs["anomaly"], "anomaly", min_value, max_value, multi_path)

        # Single plots (if data exists)
        if single_data and "single" in plot_data_cache:
            self.mono_plotter(single_data, mlip_name, "single", min_value, max_value, mono_path, plot_data_cache.get("single"))
            self.multi_plotter(single_data, mlip_name, ["all"], "single", min_value, max_value, multi_path)

        return MAE_total, MAE_normal, MAEs_total_multi, MAEs_normal_multi

    def _get_adsorbate_anomaly_breakdown(self, ads_data, adsorbate):
        """
        Get anomaly breakdown for a specific adsorbate.

        Args:
            ads_data: Dictionary containing all adsorbate data
            adsorbate: Specific adsorbate to analyze

        Returns:
            dict: Anomaly counts and rates for the adsorbate
        """
        if adsorbate not in ads_data:
            return {
                "normal_count": 0,
                "reproduction_failure_count": 0,
                "unphysical_relaxation_count": 0,
                "adsorbate_migration_count": 0,
                "energy_anomaly_count": 0,
                "total_count": 0,
                "anomaly_total": 0,
                "reproduction_failure_rate": 0.0,
                "unphysical_relaxation_rate": 0.0,
                "adsorbate_migration_rate": 0.0,
                "energy_anomaly_rate": 0.0,
                "anomaly_rate": 0.0
            }

        normal_count = len(ads_data[adsorbate]["normal"]["DFT"])
        reproduction_failure_count = len(ads_data[adsorbate]["reproduction_failure"]["DFT"])
        unphysical_relaxation_count = len(ads_data[adsorbate]["unphysical_relaxation"]["DFT"])
        adsorbate_migration_count = len(ads_data[adsorbate]["adsorbate_migration"]["DFT"])
        energy_anomaly_count = len(ads_data[adsorbate]["energy_anomaly"]["DFT"])

        # Anomaly total excludes adsorbate migration (now independent category)
        anomaly_total = (reproduction_failure_count + unphysical_relaxation_count + 
                        energy_anomaly_count)
        total_count = normal_count + anomaly_total + adsorbate_migration_count

        if total_count > 0:
            reproduction_failure_rate = (reproduction_failure_count / total_count) * 100
            unphysical_relaxation_rate = (unphysical_relaxation_count / total_count) * 100
            adsorbate_migration_rate = (adsorbate_migration_count / total_count) * 100
            energy_anomaly_rate = (energy_anomaly_count / total_count) * 100
            anomaly_rate = (anomaly_total / total_count) * 100
        else:
            reproduction_failure_rate = unphysical_relaxation_rate = 0.0
            adsorbate_migration_rate = energy_anomaly_rate = anomaly_rate = 0.0

        return {
            "normal_count": normal_count,
            "reproduction_failure_count": reproduction_failure_count,
            "unphysical_relaxation_count": unphysical_relaxation_count,
            "adsorbate_migration_count": adsorbate_migration_count,
            "energy_anomaly_count": energy_anomaly_count,
            "total_count": total_count,
            "anomaly_total": anomaly_total,
            "reproduction_failure_rate": reproduction_failure_rate,
            "unphysical_relaxation_rate": unphysical_relaxation_rate,
            "adsorbate_migration_rate": adsorbate_migration_rate,
            "energy_anomaly_rate": energy_anomaly_rate,
            "anomaly_rate": anomaly_rate
        }

    def _anomaly_detection(self, mlip_result, mlip_name, n_crit_relax):
        """
        Perform anomaly detection using direct data access for fast classification.

        This method:
        1. Loads all data once from result.json
        2. Pre-processes all reaction data once
        3. Performs only threshold comparisons for the specified values

        Returns the same format as _detect_anomalies for compatibility.
        """

        # Get chemical_bond_cutoff from calculation settings (required)
        calculation_settings = mlip_result.get("calculation_settings", {})
        if "chemical_bond_cutoff" not in calculation_settings:
            raise ValueError(f"chemical_bond_cutoff not found in calculation_settings for {mlip_name}. "
                           "Please re-run calculations with updated CatBench version.")
        chemical_bond_cutoff = calculation_settings["chemical_bond_cutoff"]

        # Initialize result structures
        anomaly_detection_result = {
            "thresholds": {
                "disp_thrs": self.disp_thrs,
                "energy_thrs": self.energy_thrs,
                "reproduction_thrs": self.reproduction_thrs,
                "chemical_bond_cutoff": chemical_bond_cutoff,
                "bond_length_change_threshold": self.bond_length_change_threshold,
                "n_crit_relax": n_crit_relax
            },
            "normal": [],
            "energy_anomaly": [],
            "adsorbate_migration": [],
            "unphysical_relaxation": [],
            "reproduction_failure": []
        }

        anomaly_summary = {}

        # Determine slab calculation mode
        has_slab_calculations = None
        for reaction in mlip_result:
            if reaction != "calculation_settings":
                calculator_keys = get_calculator_keys(mlip_result[reaction])
                if len(calculator_keys) > 0:
                    has_slab_calculations = "slab_max_disp" in mlip_result[reaction][calculator_keys[0]]
                break

        # Pre-calculate bond change threshold percentage
        bond_threshold_pct = self.bond_length_change_threshold * 100

        # Process each reaction
        for reaction in mlip_result:
            if reaction == "calculation_settings":
                continue

            # Apply energy cutoff filter (only if cutoff is specified)
            if self.energy_cutoff is not None:
                try:
                    reference_energy = mlip_result[reaction]["reference"]["ads_eng"]
                    if reference_energy > self.energy_cutoff:
                        continue  # Skip reactions with energy above cutoff
                except (KeyError, TypeError):
                    continue  # Skip if reference energy is not available

            # Get pre-calculated data
            calculator_keys = get_calculator_keys(mlip_result[reaction])
            median_calc_key = get_median_calculator_key(mlip_result[reaction])

            # Get bond change and substrate displacement directly from median calculator result
            median_data = mlip_result[reaction].get(median_calc_key, {})
            bond_change_pct = median_data.get("max_bond_change", 0.0)
            substrate_disp_median = median_data.get("substrate_displacement", 0.0)

            # Initialize anomaly counters for detailed tracking
            anomalies = {
                "slab_conv": 0,
                "ads_conv": 0,
                "slab_move": 0,
                "ads_move": 0,
                "slab_seed": 0,
                "ads_seed": 0,
                "ads_eng_seed": 0,
                "adsorbate_migration": 0,
                "energy_anomaly": 0,
            }

            # Fast convergence and displacement checks
            for calc_key in calculator_keys:
                calc_data = mlip_result[reaction][calc_key]

                # Convergence anomalies
                if has_slab_calculations and "slab_steps" in calc_data:
                    if calc_data["slab_steps"] == n_crit_relax:
                        anomalies["slab_conv"] += 1

                if "adslab_steps" in calc_data:
                    if calc_data["adslab_steps"] == n_crit_relax:
                        anomalies["ads_conv"] += 1

                # Displacement anomalies
                if has_slab_calculations and "slab_max_disp" in calc_data:
                    if calc_data["slab_max_disp"] > self.disp_thrs:
                        anomalies["slab_move"] += 1

                # Substrate displacement from result.json (only for median calculator)
                if calc_key == median_calc_key:
                    if substrate_disp_median > self.disp_thrs:
                        anomalies["ads_move"] += 1

            # Fast seed range checks
            if "final" in mlip_result[reaction]:
                final_data = mlip_result[reaction]["final"]

                if has_slab_calculations and "slab_seed_range" in final_data:
                    if final_data["slab_seed_range"] > self.reproduction_thrs:
                        anomalies["slab_seed"] = 1

                if "ads_seed_range" in final_data:
                    if final_data["ads_seed_range"] > self.reproduction_thrs:
                        anomalies["ads_seed"] = 1

                if "ads_eng_seed_range" in final_data:
                    if final_data["ads_eng_seed_range"] > self.reproduction_thrs:
                        anomalies["ads_eng_seed"] = 1

                # Energy anomaly check
                if ("ads_eng_median" in final_data and 
                    "reference" in mlip_result[reaction] and
                    "ads_eng" in mlip_result[reaction]["reference"]):

                    median_energy = final_data["ads_eng_median"]
                    reference_energy = mlip_result[reaction]["reference"]["ads_eng"]
                    energy_diff = abs(median_energy - reference_energy)

                    if energy_diff > self.energy_thrs:
                        anomalies["energy_anomaly"] = 1

            # Adsorbate migration check (only for non-unphysical reactions)
            migration_check_needed = (anomalies["slab_conv"] == 0 and anomalies["ads_conv"] == 0 and 
                                    anomalies["slab_move"] == 0 and anomalies["ads_move"] == 0)

            if migration_check_needed:
                if bond_change_pct > bond_threshold_pct:
                    anomalies["adsorbate_migration"] = 1

            # Remove slab-related anomalies for OC20 mode
            if not has_slab_calculations:
                anomalies["slab_conv"] = 0
                anomalies["slab_move"] = 0
                anomalies["slab_seed"] = 0

            # Fast classification (same priority order as original)
            if anomalies["slab_seed"] > 0 or anomalies["ads_seed"] > 0 or anomalies["ads_eng_seed"] > 0:
                classification = "reproduction_failure"
                anomaly_detection_result["reproduction_failure"].append(reaction)
            elif anomalies["slab_conv"] > 0 or anomalies["ads_conv"] > 0 or anomalies["slab_move"] > 0 or anomalies["ads_move"] > 0:
                classification = "unphysical_relaxation"
                # Record which specific unphysical issues occurred
                unphysical_issues = []
                if anomalies["slab_conv"] > 0:
                    unphysical_issues.append("slab_conv")
                if anomalies["ads_conv"] > 0:
                    unphysical_issues.append("ads_conv")
                if anomalies["slab_move"] > 0:
                    unphysical_issues.append("slab_move")
                if anomalies["ads_move"] > 0:
                    unphysical_issues.append("ads_move")

                anomaly_detection_result["unphysical_relaxation"].append({
                    "reaction": reaction,
                    "issues": unphysical_issues
                })
            elif anomalies["adsorbate_migration"] > 0:
                classification = "adsorbate_migration"
                anomaly_detection_result["adsorbate_migration"].append(reaction)
            elif anomalies["energy_anomaly"] > 0:
                classification = "energy_anomaly"
                anomaly_detection_result["energy_anomaly"].append(reaction)
            else:
                classification = "normal"
                anomaly_detection_result["normal"].append(reaction)

            anomaly_summary[reaction] = {
                "classification": classification,
                "details": anomalies
            }

        total_reactions = len(anomaly_summary)

        return anomaly_detection_result, anomaly_summary

