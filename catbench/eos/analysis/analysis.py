"""
EOS Analysis Module for CatBench.

Provides comprehensive analysis and visualization for EOS benchmarking results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scipy.optimize import curve_fit
import pandas as pd
import xlsxwriter
from catbench.utils.analysis_utils import set_matplotlib_font
from catbench.config import ANALYSIS_DEFAULTS, get_default


class EOSAnalysis:
    """
    EOS Analysis class for comprehensive MLIP benchmarking analysis.
    
    Features:
        - Individual MLIP vs VASP EOS plots
        - Multi-MLIP comparison plots
        - 2nd order polynomial fitting for equilibrium volume
        - RMSE/MAE calculation
        - Excel report generation
        
    Args:
        mlip_list (list, optional): List of MLIP names to analyze.
                                  Default: Auto-detect from result/eos/
        benchmark (str, optional): Name of the benchmark dataset.
                                 Default: Working directory name
        figsize (tuple, optional): Figure size for plots. Default: (12, 9)
        dpi (int, optional): Resolution for saved plots. Default: 300
        font_setting (tuple, optional): Font configuration (path, family)
        mlip_name_map (dict, optional): Mapping from raw MLIP names to display names for plots/Excel. Default: {} (no mapping)
    """
    
    def __init__(self, **kwargs):
        """Initialize EOS analysis configuration."""
        
        self.calculating_path = kwargs.get("calculating_path", os.path.join(os.getcwd(), "result"))
        self.plot_path = kwargs.get("plot_path", os.path.join(os.getcwd(), "plot"))
        self.benchmark = kwargs.get("benchmark", os.path.basename(os.getcwd()))
        self.mlip_list = kwargs.get("mlip_list", None)
        
        # Plot settings from config
        self.figsize = kwargs.get("figsize", get_default("figsize", ANALYSIS_DEFAULTS))
        self.dpi = kwargs.get("dpi", get_default("dpi", ANALYSIS_DEFAULTS))
        self.font_setting = kwargs.get("font_setting", get_default("font_setting", ANALYSIS_DEFAULTS))
        
        # Tick control settings
        self.x_tick_bins = kwargs.get("x_tick_bins", get_default("x_tick_bins", ANALYSIS_DEFAULTS))
        self.y_tick_bins = kwargs.get("y_tick_bins", get_default("y_tick_bins", ANALYSIS_DEFAULTS))
        self.tick_decimal_places = kwargs.get("tick_decimal_places", get_default("tick_decimal_places", ANALYSIS_DEFAULTS))
        self.tick_labelsize = kwargs.get("tick_labelsize", get_default("tick_labelsize", ANALYSIS_DEFAULTS))
        
        # Font settings
        self.legend_fontsize = kwargs.get("legend_fontsize", get_default("legend_fontsize", ANALYSIS_DEFAULTS))
        self.xlabel_fontsize = kwargs.get("xlabel_fontsize", get_default("xlabel_fontsize", ANALYSIS_DEFAULTS))
        self.ylabel_fontsize = kwargs.get("ylabel_fontsize", get_default("ylabel_fontsize", ANALYSIS_DEFAULTS))
        self.comparison_legend_fontsize = kwargs.get("comparison_legend_fontsize", get_default("comparison_legend_fontsize", ANALYSIS_DEFAULTS))
        
        # Marker settings
        self.mark_size = kwargs.get("mark_size", get_default("mark_size", ANALYSIS_DEFAULTS))
        
        # Grid setting
        self.grid = kwargs.get("grid", get_default("grid", ANALYSIS_DEFAULTS))
        # Display name mapping for MLIP names (for plots/Excel display only)
        self.mlip_name_map = kwargs.get("mlip_name_map", get_default("mlip_name_map", ANALYSIS_DEFAULTS)) or {}
        
        # Auto-detect MLIPs if not provided
        if self.mlip_list is None:
            self.mlip_list = self._auto_detect_mlips()
        
        # Create consistent color mapping for MLIPs
        self.mlip_colors = self._create_mlip_color_mapping()
        
        print(f"Benchmark: {self.benchmark}")
        print(f"MLIPs to analyze: {self.mlip_list}")
    
    def _display_mlip_name(self, mlip_name):
        """Return mapped display name for MLIP if provided, else original name."""
        return self.mlip_name_map.get(mlip_name, mlip_name)
    
    def _create_mlip_color_mapping(self):
        """Create consistent color mapping for MLIPs."""
        # MLIP color palette - distinct and colorblind-friendly
        mlip_colors_palette = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange  
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # olive
            '#17becf',  # cyan
            '#aec7e8',  # light blue
            '#ffbb78',  # light orange
            '#98df8a',  # light green
            '#ff9896',  # light red
            '#c5b0d5',  # light purple
        ]
        
        mlip_colors = {}
        for i, mlip_name in enumerate(sorted(self.mlip_list, key=str.lower)):
            mlip_colors[mlip_name] = mlip_colors_palette[i % len(mlip_colors_palette)]
        
        return mlip_colors
    
    def _auto_detect_mlips(self):
        """Auto-detect available MLIP results."""
        if not os.path.exists(self.calculating_path):
            return []
        
        mlip_dirs = []
        for item in os.listdir(self.calculating_path):
            item_path = os.path.join(self.calculating_path, item)
            if os.path.isdir(item_path):
                # Check if result file exists
                result_file = os.path.join(item_path, f"{item}_eos_result.json")
                if os.path.exists(result_file):
                    mlip_dirs.append(item)
        
        return sorted(mlip_dirs, key=str.lower)
    
    def analysis(self):
        """Run complete EOS analysis."""
        print("\n" + "="*60)
        print("Starting EOS Analysis")
        print("="*60)
        
        if not self.mlip_list:
            print("No MLIP results found!")
            return
        
        # Load all results
        all_results = self._load_all_results()
        
        # Get list of all materials
        materials = set()
        for mlip_results in all_results.values():
            materials.update(mlip_results["results"].keys())
        materials = sorted(list(materials))
        
        print(f"Found {len(materials)} materials: {materials}")
        
        # 1. Individual MLIP plots (saved in each MLIP folder)
        print("\n--- Creating individual MLIP plots ---")
        for mlip_name in sorted(self.mlip_list, key=str.lower):
            if mlip_name in all_results:
                self._create_individual_plots(mlip_name, all_results[mlip_name], materials)
        
        # 2. Multi-MLIP comparison plots (saved in plot/)
        print("\n--- Creating multi-MLIP comparison plots ---")
        self._create_comparison_plots(all_results, materials)
        
        # 3. Calculate metrics and create Excel report
        print("\n--- Calculating metrics and creating Excel report ---")
        self._create_excel_report(all_results, materials)
        
        print("\n" + "="*60)
        print("EOS Analysis Complete!")
        print("="*60)
    
    def _load_all_results(self):
        """Load all MLIP results."""
        all_results = {}
        
        for mlip_name in self.mlip_list:
            result_file = os.path.join(self.calculating_path, mlip_name, f"{mlip_name}_eos_result.json")
            
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    all_results[mlip_name] = json.load(f)
            else:
                print(f"Warning: Result file not found for {mlip_name}")
        
        return all_results
    
    def _create_individual_plots(self, mlip_name, mlip_results, materials):
        """Create individual MLIP vs VASP plots."""
        display_name = self._display_mlip_name(mlip_name)
        print(f"  Creating plots for {display_name}...")
        
        # Create plot directory in plot/MLIP_name/
        plot_dir = os.path.join(self.plot_path, display_name)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Setup font if provided
        if self.font_setting:
            set_matplotlib_font(self.font_setting[0], self.font_setting[1])
        
        # Create plot for each material
        for material in materials:
            if material not in mlip_results["results"]:
                continue
            
            material_data = mlip_results["results"][material]
            
            # Extract data
            volumes = []
            vasp_energies = []
            mlip_energies = []
            
            for point in material_data["points"]:
                if point["mlip_energy"] is not None:
                    volumes.append(point["volume"])
                    vasp_energies.append(point["vasp_energy"])
                    mlip_energies.append(point["mlip_energy"])
            
            if len(volumes) < 2:
                print(f"    Skipping {material}: insufficient data points")
                continue
            
            # Convert to numpy arrays
            volumes = np.array(volumes)
            vasp_energies = np.array(vasp_energies)
            mlip_energies = np.array(mlip_energies)
            
            # Calculate relative energies
            vasp_rel = vasp_energies - np.min(vasp_energies)
            mlip_rel = mlip_energies - np.min(mlip_energies)
            
            # Fit Birch-Murnaghan EOS
            vasp_fit_params = self._fit_birch_murnaghan(volumes, vasp_energies)
            mlip_fit_params = self._fit_birch_murnaghan(volumes, mlip_energies)
            
            # Create plot with config settings
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Apply font settings if configured
            if self.font_setting:
                set_matplotlib_font(*self.font_setting)
            
            # Plot data points with MLIP colors
            ax.plot(volumes, vasp_rel, 'o', color='black', label='VASP', markersize=self.mark_size/10)
            ax.plot(volumes, mlip_rel, 's', color=self.mlip_colors[mlip_name], label=display_name, markersize=self.mark_size/10)
            
            # Plot fitted curves
            if vasp_fit_params is not None:
                v_fit = np.linspace(np.min(volumes), np.max(volumes), 100)
                vasp_fit = self._birch_murnaghan_eos(v_fit, *vasp_fit_params)
                # Convert to relative energies for plotting
                vasp_fit_rel = vasp_fit - np.min(vasp_fit)
                ax.plot(v_fit, vasp_fit_rel, '-', color='black', alpha=0.5, linewidth=2)
                
                # Mark equilibrium volume from fit
                v_eq_vasp = vasp_fit_params[1]  # V0 parameter
                ax.axvline(v_eq_vasp, color='black', linestyle='--', alpha=0.3)
            
            if mlip_fit_params is not None:
                v_fit = np.linspace(np.min(volumes), np.max(volumes), 100)
                mlip_fit = self._birch_murnaghan_eos(v_fit, *mlip_fit_params)
                # Convert to relative energies for plotting
                mlip_fit_rel = mlip_fit - np.min(mlip_fit)
                ax.plot(v_fit, mlip_fit_rel, '-', color=self.mlip_colors[mlip_name], alpha=0.5, linewidth=2)
                
                # Mark equilibrium volume from fit
                v_eq_mlip = mlip_fit_params[1]  # V0 parameter
                ax.axvline(v_eq_mlip, color=self.mlip_colors[mlip_name], linestyle='--', alpha=0.3)
            
            # Apply config-based styling
            ax.set_xlabel('Volume (Å³)', fontsize=self.xlabel_fontsize, fontweight='bold')
            ax.set_ylabel('Relative Energy (eV)', fontsize=self.ylabel_fontsize, fontweight='bold')
            # Title removed as requested
            
            # Tick control from config
            if self.x_tick_bins is not None:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=self.x_tick_bins))
            if self.y_tick_bins is not None:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=self.y_tick_bins))
            if self.tick_decimal_places is not None:
                fmt = f"%.{self.tick_decimal_places}f"
                ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
                ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
            
            ax.tick_params(axis="both", which="major", labelsize=self.tick_labelsize)
            ax.legend(fontsize=self.legend_fontsize, frameon=True, fancybox=True, shadow=False)
            
            # Grid from config
            if self.grid:
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(plot_dir, f"EOS_{material}.png")
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved: {plot_file}")
    
    def _create_comparison_plots(self, all_results, materials):
        """Create multi-MLIP comparison plots."""
        print("  Creating comparison plots...")
        
        # Create plot directory (directly in plot/)
        os.makedirs(self.plot_path, exist_ok=True)
        
        # Use MLIP color mapping for consistency
        
        # Create plot for each material
        for material in materials:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Apply font settings if configured
            if self.font_setting:
                set_matplotlib_font(*self.font_setting)
            
            # First plot VASP reference
            vasp_plotted = False
            
            for idx, mlip_name in enumerate(sorted(self.mlip_list, key=str.lower)):
                if mlip_name not in all_results:
                    continue
                
                mlip_results = all_results[mlip_name]
                if material not in mlip_results["results"]:
                    continue
                
                material_data = mlip_results["results"][material]
                
                # Extract data
                volumes = []
                vasp_energies = []
                mlip_energies = []
                
                for point in material_data["points"]:
                    if point["mlip_energy"] is not None:
                        volumes.append(point["volume"])
                        vasp_energies.append(point["vasp_energy"])
                        mlip_energies.append(point["mlip_energy"])
                
                if len(volumes) < 2:
                    continue
                
                # Convert to numpy arrays
                volumes = np.array(volumes)
                vasp_energies = np.array(vasp_energies)
                mlip_energies = np.array(mlip_energies)
                
                # Calculate relative energies
                vasp_rel = vasp_energies - np.min(vasp_energies)
                mlip_rel = mlip_energies - np.min(mlip_energies)
                
                # Plot VASP only once
                if not vasp_plotted:
                    ax.plot(volumes, vasp_rel, 'o-', color='black', 
                           label='VASP', markersize=self.mark_size/10, linewidth=3, alpha=0.9)
                    vasp_plotted = True
                
                # Plot MLIP with consistent colors
                display_name = self._display_mlip_name(mlip_name)
                ax.plot(volumes, mlip_rel, 'o-', color=self.mlip_colors[mlip_name], 
                       label=display_name, markersize=self.mark_size/12, linewidth=2, alpha=0.7)
            
            if not vasp_plotted:
                print(f"    Skipping {material}: no valid data")
                plt.close()
                continue
            
            # Apply config-based styling
            ax.set_xlabel('Volume (Å³)', fontsize=self.xlabel_fontsize, fontweight='bold')
            ax.set_ylabel('Relative Energy (eV)', fontsize=self.ylabel_fontsize, fontweight='bold')
            # Title removed as requested
            
            # Tick control from config
            if self.x_tick_bins is not None:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=self.x_tick_bins))
            if self.y_tick_bins is not None:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=self.y_tick_bins))
            if self.tick_decimal_places is not None:
                fmt = f"%.{self.tick_decimal_places}f"
                ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
                ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
            
            ax.tick_params(axis="both", which="major", labelsize=self.tick_labelsize)
            ax.legend(fontsize=self.comparison_legend_fontsize, frameon=True, fancybox=True, shadow=False, ncol=2, loc='upper right')
            
            # Grid from config
            if self.grid:
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot directly in plot/
            plot_file = os.path.join(self.plot_path, f"EOS_comparison_{material}.png")
            plt.savefig(plot_file, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved: {plot_file}")
    
    def _create_excel_report(self, all_results, materials):
        """Create comprehensive Excel report with multiple sheets."""
        excel_file = os.path.join(os.getcwd(), f"{self.benchmark}_EOS_Analysis.xlsx")
        
        # Prepare data for sheets
        material_comparison_data = {}
        mlip_comparison_data = {}
        
        for material in materials:
            for mlip_name in sorted(self.mlip_list, key=str.lower):
                if mlip_name not in all_results:
                    continue
                
                mlip_results = all_results[mlip_name]
                if material not in mlip_results["results"]:
                    continue
                
                material_data = mlip_results["results"][material]
                
                # Calculate metrics using relative energies
                rmse, mae = self._calculate_metrics(material_data)
                
                # Get equilibrium volumes from data points
                vasp_eq_vol_data = material_data.get("vasp_equilibrium_volume", None)
                mlip_eq_vol_data = material_data.get("mlip_equilibrium_volume", None)
                
                # Extract data for fitting
                volumes = []
                vasp_energies = []
                mlip_energies = []
                
                for point in material_data["points"]:
                    if point["mlip_energy"] is not None:
                        volumes.append(point["volume"])
                        vasp_energies.append(point["vasp_energy"])
                        mlip_energies.append(point["mlip_energy"])
                
                # Fit Birch-Murnaghan EOS to get EOS parameters
                vasp_eos_params = None
                mlip_eos_params = None
                
                if len(volumes) >= 4:  # Need at least 4 points for Birch-Murnaghan EOS fitting
                    volumes = np.array(volumes)
                    vasp_energies = np.array(vasp_energies)
                    mlip_energies = np.array(mlip_energies)
                    
                    # Fit Birch-Murnaghan EOS: [E0, V0, B0, B0_prime]
                    vasp_eos_params = self._fit_birch_murnaghan(volumes, vasp_energies)
                    mlip_eos_params = self._fit_birch_murnaghan(volumes, mlip_energies)
                
                # Extract EOS parameters
                vasp_E0, vasp_V0, vasp_B0, vasp_B0_prime = (None, None, None, None)
                mlip_E0, mlip_V0, mlip_B0, mlip_B0_prime = (None, None, None, None)
                
                if vasp_eos_params is not None:
                    vasp_E0, vasp_V0, vasp_B0, vasp_B0_prime = vasp_eos_params
                    
                if mlip_eos_params is not None:
                    mlip_E0, mlip_V0, mlip_B0, mlip_B0_prime = mlip_eos_params
                
                # Calculate parameter differences (excluding E0 - functional difference)
                vol_error_eos = None
                B0_error = None
                B0_prime_error = None
                
                if vasp_V0 and mlip_V0:
                    vol_error_eos = abs(mlip_V0 - vasp_V0)
                    
                if vasp_B0 and mlip_B0:
                    B0_error = abs(mlip_B0 - vasp_B0)
                    
                if vasp_B0_prime and mlip_B0_prime:
                    B0_prime_error = abs(mlip_B0_prime - vasp_B0_prime)
                
                # Data for material comparison sheets (more comprehensive)
                if material not in material_comparison_data:
                    material_comparison_data[material] = []
                material_comparison_data[material].append({
                    "MLIP": self._display_mlip_name(mlip_name),
                    "RMSE (rel. eV)": rmse,
                    "MAE (rel. eV)": mae,
                    "VASP V0 [BM] (Å³)": vasp_V0,
                    "MLIP V0 [BM] (Å³)": mlip_V0,
                    "V0 Error (Å³)": vol_error_eos,
                    "VASP B0 [BM] (GPa)": vasp_B0,
                    "MLIP B0 [BM] (GPa)": mlip_B0,
                    "B0 Error (GPa)": B0_error,
                    "VASP B0' [BM]": vasp_B0_prime,
                    "MLIP B0' [BM]": mlip_B0_prime,
                    "B0' Error": B0_prime_error,
                    "N Points": material_data.get("n_valid_points", 0)
                })
                
                # Data for MLIP comparison sheets (more comprehensive)
                if mlip_name not in mlip_comparison_data:
                    mlip_comparison_data[mlip_name] = []
                mlip_comparison_data[mlip_name].append({
                    "Material": material,
                    "RMSE (rel. eV)": rmse,
                    "MAE (rel. eV)": mae,
                    "VASP V0 [BM] (Å³)": vasp_V0,
                    "MLIP V0 [BM] (Å³)": mlip_V0,
                    "V0 Error (Å³)": vol_error_eos,
                    "VASP B0 [BM] (GPa)": vasp_B0,
                    "MLIP B0 [BM] (GPa)": mlip_B0,
                    "B0 Error (GPa)": B0_error,
                    "VASP B0' [BM]": vasp_B0_prime,
                    "MLIP B0' [BM]": mlip_B0_prime,
                    "B0' Error": B0_prime_error,
                    "N Points": material_data.get("n_valid_points", 0)
                })
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats (AdsorptionAnalysis style - no background color)
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })
            center_format = workbook.add_format({
                'align': 'center',
                'valign': 'vcenter'
            })
            number_format_1f = workbook.add_format({
                'num_format': '0.0',
                'align': 'center',
                'valign': 'vcenter'
            })
            number_format_2f = workbook.add_format({
                'num_format': '0.00',
                'align': 'center',
                'valign': 'vcenter'
            })
            number_format_3f = workbook.add_format({
                'num_format': '0.000',
                'align': 'center',
                'valign': 'vcenter'
            })
            number_format_4f = workbook.add_format({
                'num_format': '0.0000',
                'align': 'center',
                'valign': 'vcenter'
            })
            number_format_int = workbook.add_format({
                'num_format': '0',
                'align': 'center',
                'valign': 'vcenter'
            })
            
            # 1. Create Summary sheet first
            self._create_summary_sheet(writer, all_results, materials, header_format, 
                                     center_format, number_format_3f, number_format_int)
            
            # 2. Material comparison sheets (each material gets its own sheet)
            for material in sorted(material_comparison_data.keys()):
                material_data = material_comparison_data[material]
                df_material = pd.DataFrame(material_data)
                if not df_material.empty:
                    # Sort by MLIP name alphabetically (case-insensitive)
                    df_material['sort_key'] = df_material['MLIP'].str.lower()
                    df_material = df_material.sort_values('sort_key')
                    df_material = df_material.drop('sort_key', axis=1)
                    sheet_name = f'{material}'  # Simple sheet name
                    df_material.to_excel(writer, sheet_name=sheet_name, index=False)
                    worksheet = writer.sheets[sheet_name]
                    self._format_worksheet_detailed(worksheet, df_material, header_format,
                                                   center_format, number_format_3f,
                                                   number_format_1f, number_format_int)
            
            # 3. MLIP comparison sheets (each MLIP gets its own sheet)
            for mlip_name in sorted(mlip_comparison_data.keys(), key=str.lower):
                mlip_data = mlip_comparison_data[mlip_name]
                df_mlip = pd.DataFrame(mlip_data)
                if not df_mlip.empty:
                    # Sort by material name alphabetically
                    df_mlip = df_mlip.sort_values('Material')
                    sheet_name = f'{self._display_mlip_name(mlip_name)}'  # Simple sheet name
                    df_mlip.to_excel(writer, sheet_name=sheet_name, index=False)
                    worksheet = writer.sheets[sheet_name]
                    self._format_worksheet_detailed(worksheet, df_mlip, header_format,
                                                   center_format, number_format_3f,
                                                   number_format_1f, number_format_int)
            
            # Overall comparison sheet removed - only Material and MLIP sheets remain
        
        print(f"Excel report saved to: {excel_file}")
    
    def _create_summary_sheet(self, writer, all_results, materials, header_format, 
                             center_format, number_format, int_format):
        """Create Summary sheet with overall MLIP performance metrics."""
        summary_data = []
        
        for mlip_name in sorted(self.mlip_list, key=str.lower):
            if mlip_name not in all_results:
                continue
                
            mlip_results = all_results[mlip_name]
            
            # Collect all data points for overall RMSE/MAE calculation
            all_vasp_energies = []
            all_mlip_energies = []
            
            # Collect EOS parameter errors for averaging (excluding E0 - functional difference)
            V0_errors = []
            B0_errors = []
            B0_prime_errors = []
            
            n_systems = 0
            
            for material in materials:
                if material not in mlip_results["results"]:
                    continue
                    
                material_data = mlip_results["results"][material]
                n_systems += 1
                
                # Collect energy data for overall metrics
                volumes = []
                vasp_energies = []
                mlip_energies = []
                
                for point in material_data["points"]:
                    if point["mlip_energy"] is not None:
                        volumes.append(point["volume"])
                        vasp_energies.append(point["vasp_energy"])
                        mlip_energies.append(point["mlip_energy"])
                
                if len(volumes) >= 4:  # Need enough points for BM EOS
                    # Add to overall energy collections
                    all_vasp_energies.extend(vasp_energies)
                    all_mlip_energies.extend(mlip_energies)
                    
                    # Fit Birch-Murnaghan EOS for parameter errors
                    volumes = np.array(volumes)
                    vasp_energies = np.array(vasp_energies)
                    mlip_energies = np.array(mlip_energies)
                    
                    vasp_eos_params = self._fit_birch_murnaghan(volumes, vasp_energies)
                    mlip_eos_params = self._fit_birch_murnaghan(volumes, mlip_energies)
                    
                    if vasp_eos_params is not None and mlip_eos_params is not None:
                        vasp_E0, vasp_V0, vasp_B0, vasp_B0_prime = vasp_eos_params
                        mlip_E0, mlip_V0, mlip_B0, mlip_B0_prime = mlip_eos_params
                        
                        V0_errors.append(abs(mlip_V0 - vasp_V0))
                        B0_errors.append(abs(mlip_B0 - vasp_B0))
                        B0_prime_errors.append(abs(mlip_B0_prime - vasp_B0_prime))
            
            # Calculate overall RMSE and MAE
            overall_rmse = None
            overall_mae = None
            
            if len(all_vasp_energies) > 0:
                all_vasp_energies = np.array(all_vasp_energies)
                all_mlip_energies = np.array(all_mlip_energies)
                
                # Use relative energies for metrics
                vasp_rel = all_vasp_energies - np.min(all_vasp_energies)
                mlip_rel = all_mlip_energies - np.min(all_mlip_energies)
                
                overall_rmse = np.sqrt(np.mean((vasp_rel - mlip_rel)**2))
                overall_mae = np.mean(np.abs(vasp_rel - mlip_rel))
            
            # Calculate average parameter errors (excluding E0 - functional difference)
            avg_V0_error = np.mean(V0_errors) if V0_errors else None
            avg_B0_error = np.mean(B0_errors) if B0_errors else None
            avg_B0_prime_error = np.mean(B0_prime_errors) if B0_prime_errors else None
            
            summary_data.append({
                "MLIP": self._display_mlip_name(mlip_name),
                "N Systems": n_systems,
                "Overall RMSE (rel. eV)": overall_rmse,
                "Overall MAE (rel. eV)": overall_mae,
                "Avg V0 Error (Å³)": avg_V0_error,
                "Avg B0 Error (GPa)": avg_B0_error,
                "Avg B0' Error": avg_B0_prime_error,
            })
        
        # Create DataFrame and write to Summary sheet
        df_summary = pd.DataFrame(summary_data)
        if not df_summary.empty:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            worksheet = writer.sheets['Summary']
            self._format_worksheet_detailed(worksheet, df_summary, header_format,
                                           center_format, number_format,
                                           number_format, int_format)
    
    def _format_worksheet_detailed(self, worksheet, df, header_format, center_format,
                                  number_format, volume_format, int_format):
        """Format worksheet with proper alignment and specific number formatting."""
        # Write headers with format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Apply specific formats to data cells based on column type
        for row_idx in range(1, len(df) + 1):
            for col_idx, col_name in enumerate(df.columns):
                value = df.iloc[row_idx - 1, col_idx]
                
                if pd.isna(value):
                    worksheet.write(row_idx, col_idx, "", center_format)
                elif 'N Points' in col_name:
                    worksheet.write(row_idx, col_idx, value, int_format)
                elif 'Vol' in col_name or 'Å³' in col_name:
                    worksheet.write(row_idx, col_idx, value, volume_format)
                elif col_name in ['MLIP', 'Material']:
                    worksheet.write(row_idx, col_idx, value, center_format)
                elif isinstance(value, (int, float)):
                    worksheet.write(row_idx, col_idx, value, number_format)
                else:
                    worksheet.write(row_idx, col_idx, value, center_format)
        
        # Auto-adjust column widths with specific sizing for different column types
        for i, col in enumerate(df.columns):
            if col in ['MLIP', 'Material']:
                # Wider columns for MLIP and Material names
                column_len = df[col].astype(str).str.len().max()
                column_len = max(column_len, len(col)) + 4  # Extra padding for names
                worksheet.set_column(i, i, min(column_len, 30))  # Max width 30
            elif any(keyword in col for keyword in ['Overall RMSE', 'Overall MAE', 'Avg V0 Error', 'Avg B0 Error', 'Avg B0\' Error']):
                # Extra wide columns for summary metrics with long headers
                column_len = df[col].astype(str).str.len().max()
                column_len = max(column_len, len(col)) + 4  # More padding for long headers
                worksheet.set_column(i, i, min(column_len, 21))  # Max width 21 (18+3)
            else:
                # Wider columns for numeric values (increased by 3)
                column_len = df[col].astype(str).str.len().max()
                column_len = max(column_len, len(col)) + 4  # More padding for numbers
                worksheet.set_column(i, i, min(column_len, 18))  # Max width 18 (15+3)
    
    
    def _fit_birch_murnaghan(self, volumes, energies):
        """Fit 3rd order Birch-Murnaghan EOS to data."""
        try:
            # Estimate initial parameters
            initial_params = self._estimate_initial_parameters(volumes, energies)
            
            # Fit Birch-Murnaghan EOS: E(V) = E0 + (9*V0*B0/16) * {[(V0/V)^(2/3) - 1]^3 * B0' + [(V0/V)^(2/3) - 1]^2 * [6 - 4*(V0/V)^(2/3)]}
            params, _ = curve_fit(self._birch_murnaghan_eos, volumes, energies, 
                                 p0=initial_params, maxfev=5000)
            return params
        except Exception as e:
            print(f"    Warning: Birch-Murnaghan fitting failed: {e}")
            return None
    
    def _birch_murnaghan_eos(self, V, E0, V0, B0, B0_prime):
        """3rd order Birch-Murnaghan equation of state.
        
        Parameters:
        V: Volume
        E0: Equilibrium energy
        V0: Equilibrium volume
        B0: Bulk modulus (in GPa)
        B0_prime: Pressure derivative of bulk modulus
        
        Returns:
        Energy
        """
        # Convert B0 from GPa to eV/Å³ (1 GPa = 0.00624151 eV/Å³)
        B0_eV = B0 * 0.00624151
        
        eta = (V0 / V) ** (2.0/3.0)
        
        # 3rd order Birch-Murnaghan EOS
        energy = E0 + (9.0 * V0 * B0_eV / 16.0) * (
            (eta - 1.0) ** 3 * B0_prime + 
            (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
        )
        
        return energy
    
    def _estimate_initial_parameters(self, volumes, energies):
        """Estimate initial parameters for Birch-Murnaghan EOS fitting.
        
        Returns:
        [E0, V0, B0, B0_prime] where:
        - E0: minimum energy
        - V0: volume at minimum energy
        - B0: bulk modulus estimate (GPa)
        - B0_prime: pressure derivative estimate
        """
        # Find minimum energy and corresponding volume
        min_idx = np.argmin(energies)
        E0_est = energies[min_idx]
        V0_est = volumes[min_idx]
        
        # Estimate bulk modulus from curvature around minimum
        # B0 = V0 * d²E/dV² 
        # Use finite differences if we have enough points
        if len(volumes) >= 3:
            # Sort by volume
            sorted_indices = np.argsort(volumes)
            V_sorted = volumes[sorted_indices]
            E_sorted = energies[sorted_indices]
            
            # Find points around equilibrium for curvature estimation
            try:
                # Second derivative approximation
                dV = np.mean(np.diff(V_sorted))
                if dV > 0:
                    d2E_dV2 = np.gradient(np.gradient(E_sorted, V_sorted), V_sorted)[min_idx]
                    B0_est = V0_est * d2E_dV2 / 0.00624151  # Convert to GPa
                    B0_est = max(50, min(500, abs(B0_est)))  # Reasonable bounds
                else:
                    B0_est = 150.0  # Default value
            except:
                B0_est = 150.0  # Default bulk modulus in GPa
        else:
            B0_est = 150.0  # Default bulk modulus in GPa
        
        # Default B0_prime (typical value for most materials)
        B0_prime_est = 4.0
        
        return [E0_est, V0_est, B0_est, B0_prime_est]
    
    def _calculate_metrics(self, material_data):
        """Calculate RMSE and MAE between VASP and MLIP."""
        vasp_energies = []
        mlip_energies = []
        
        for point in material_data["points"]:
            if point["mlip_energy"] is not None:
                vasp_energies.append(point["vasp_energy"])
                mlip_energies.append(point["mlip_energy"])
        
        if len(vasp_energies) < 2:
            return None, None
        
        vasp_energies = np.array(vasp_energies)
        mlip_energies = np.array(mlip_energies)
        
        # Use relative energies for metrics
        vasp_rel = vasp_energies - np.min(vasp_energies)
        mlip_rel = mlip_energies - np.min(mlip_energies)
        
        rmse = np.sqrt(np.mean((vasp_rel - mlip_rel)**2))
        mae = np.mean(np.abs(vasp_rel - mlip_rel))
        
        return rmse, mae