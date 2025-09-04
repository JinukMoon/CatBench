"""
Relative Energy Analysis: Comprehensive MLIP benchmarking analysis for relative energy tasks.

Provides professional analysis and reporting for surface energy, bulk formation energy,
and custom relative energy benchmarking results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter

from catbench.utils.analysis_utils import set_matplotlib_font
from catbench.config import RELATIVE_ANALYSIS_DEFAULTS, get_default


class RelativeEnergyAnalysis:
    """
    Relative Energy Analysis class for comprehensive MLIP benchmarking analysis.
    
    Provides analysis of Machine Learning Interatomic Potential (MLIP) 
    benchmarking results for relative energy tasks.
    
    Args:
        calculating_path (str, optional): Path to calculation results directory.
                                        Default: "./result"
        plot_path (str, optional): Path to save plots.
                                 Default: "./plot_relative"  
        benchmark (str, optional): Name of the benchmark dataset.
                                 Default: Working directory name
        task_type (str): Type of relative energy task ("surface", "bulk_formation", "custom")
        figsize (tuple, optional): Figure size for plots. Default: (9, 8)
        dpi (int, optional): Resolution for saved plots. Default: 300
        font_setting (tuple, optional): Font configuration (path, family)
        specific_color (str, optional): Plot color. Default: "black"
        mark_size (int, optional): Marker size for plots. Default: 100
        linewidths (float, optional): Line width for plots. Default: 1.5
        min/max (float, optional): Manual axis limits for plots
        grid (bool, optional): Show grid on parity plots. Default: False
        mlip_name_map (dict, optional): Mapping from raw MLIP names to display names for plots/Excel. Default: {} (no mapping)
    """
    
    def __init__(self, **kwargs):
        """Initialize relative energy analysis configuration."""
        self.calculating_path = kwargs.get("calculating_path", os.path.join(os.getcwd(), "result"))
        self.plot_path = kwargs.get("plot_path", get_default("plot_path", RELATIVE_ANALYSIS_DEFAULTS))
        self.benchmark = kwargs.get("benchmark", os.path.basename(os.getcwd()))
        self.task_type = kwargs.get("task_type", None)
        self.mlip_list = kwargs.get("mlip_list", None)
        
        if not self.task_type:
            raise ValueError("task_type parameter is required")
        if self.task_type not in ["surface", "bulk_formation", "custom"]:
            raise ValueError("task_type must be one of: surface, bulk_formation, custom")
            
        # Results are now in result/MLIP_name/ not result/task_type/MLIP_name/
        self.task_result_path = self.calculating_path
        
        self.figsize = kwargs.get("figsize", get_default("figsize", RELATIVE_ANALYSIS_DEFAULTS))
        self.mark_size = kwargs.get("mark_size", get_default("mark_size", RELATIVE_ANALYSIS_DEFAULTS))
        self.linewidths = kwargs.get("linewidths", get_default("linewidths", RELATIVE_ANALYSIS_DEFAULTS))
        self.dpi = kwargs.get("dpi", get_default("dpi", RELATIVE_ANALYSIS_DEFAULTS))
        self.font_setting = kwargs.get("font_setting", get_default("font_setting", RELATIVE_ANALYSIS_DEFAULTS))
        self.specific_color = kwargs.get("specific_color", get_default("specific_color", RELATIVE_ANALYSIS_DEFAULTS))
        self.min_value = kwargs.get("min", get_default("min", RELATIVE_ANALYSIS_DEFAULTS))
        self.max_value = kwargs.get("max", get_default("max", RELATIVE_ANALYSIS_DEFAULTS))
        self.grid = kwargs.get("grid", get_default("grid", RELATIVE_ANALYSIS_DEFAULTS))
        # Display name mapping for MLIP names (for plots/Excel display only)
        self.mlip_name_map = kwargs.get("mlip_name_map", get_default("mlip_name_map", RELATIVE_ANALYSIS_DEFAULTS)) or {}
        
        self.task_info = self._get_task_info()
        
        if self.mlip_list is None:
            self.mlip_list = self._auto_detect_mlips()
        
        print(f"Configured for {self.task_info['name']} analysis")
        print(f"Benchmark: {self.benchmark}")
        print(f"Found MLIPs: {self.mlip_list}")
        print(f"Input path: {self.task_result_path}")
        print(f"Output path: {self.plot_path}")
    
    def _display_mlip_name(self, mlip_name):
        """Return mapped display name for MLIP if provided, else original name."""
        return self.mlip_name_map.get(mlip_name, mlip_name)
    
    def _get_task_info(self):
        """Get task-specific information."""
        task_info = {
            "surface": {
                "name": "Surface Energy",
                "unit": "J/m²",
                "ref_key": "surface_energy", 
                "pred_key": "surface_energy"
            },
            "bulk_formation": {
                "name": "Bulk Formation Energy",
                "unit": "eV/atom",
                "ref_key": "formation_energy",
                "pred_key": "formation_energy"
            },
            "custom": {
                "name": "Custom Relative Energy", 
                "unit": "eV",
                "ref_key": "custom_energy",
                "pred_key": "custom_energy"
            }
        }
        return task_info[self.task_type]
    
    def _auto_detect_mlips(self):
        """Auto-detect MLIP names from result directories."""
        if not os.path.exists(self.task_result_path):
            raise FileNotFoundError(f"Task result directory not found: {self.task_result_path}")
            
        mlip_dirs = []
        for item in os.listdir(self.task_result_path):
            item_path = os.path.join(self.task_result_path, item)
            if os.path.isdir(item_path):
                mlip_dirs.append(item)
         
        if not mlip_dirs:
            raise FileNotFoundError(f"No MLIP directories found in {self.task_result_path}")
             
        return sorted(mlip_dirs, key=str.lower)

    def _setup_plot(self, mlip_name):
        """Setup plot directory and font configuration."""
        display_name = self._display_mlip_name(mlip_name)
        plot_save_path = os.path.join(self.plot_path, display_name)
        os.makedirs(plot_save_path, exist_ok=True)
        
        if self.font_setting:
            set_matplotlib_font(self.font_setting[0], self.font_setting[1])
            
        return plot_save_path
    
    def _create_base_plot(self, min_value, max_value):
        """Create base plot configuration."""
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(min_value, max_value)
        ax.set_ylim(min_value, max_value)
        ax.plot([min_value, max_value], [min_value, max_value], "r-")
        if self.grid:
            ax.grid(True)
        else:
            ax.grid(False)
        
        for spine in ax.spines.values():
            spine.set_linewidth(3)
            
        return fig, ax

    def _generate_parity_plots(self):
        """Generate parity plots for each MLIP."""
        print("Generating parity plots...")
        
        all_metrics = {}
        
        for mlip_name in self.mlip_list:
            print(f"  Processing {self._display_mlip_name(mlip_name)}...")
            
            plot_save_path = self._setup_plot(mlip_name)
            
            if self.task_type == "surface":
                result_file = f"{mlip_name}_surface_energy_result.json"
            elif self.task_type == "bulk_formation":
                result_file = f"{mlip_name}_bulk_formation_result.json"
            else:  # custom
                result_file = f"{mlip_name}_custom_result.json"
                
            result_path = os.path.join(self.task_result_path, mlip_name, result_file)
            
            if not os.path.exists(result_path):
                print(f"    Warning: Result file not found: {result_path}")
                continue
                
            with open(result_path, 'r') as f:
                results = json.load(f)
            
            ref_values = []
            pred_values = []
            system_names = []
            
            for system_name, system_result in results["results"].items():                    
                try:
                    ref_value = system_result["reference"][self.task_info["ref_key"]]
                    pred_value = system_result["mlip_calculation"][self.task_info["pred_key"]]
                    
                    if self.task_type == "surface":
                        ref_value *= 16.022
                        pred_value *= 16.022
                    
                    ref_values.append(ref_value)
                    pred_values.append(pred_value)
                    system_names.append(system_name)
                    
                except KeyError as e:
                    print(f"    Warning: Missing key {e} for system {system_name}")
                    continue
            
            if not ref_values:
                print(f"    Warning: No valid data found for {mlip_name}")
                continue
            
            ref_values = np.array(ref_values)
            pred_values = np.array(pred_values)
            
            if self.min_value is not None and self.max_value is not None:
                min_val, max_val = self.min_value, self.max_value
            else:
                all_values = np.concatenate([ref_values, pred_values])
                min_val = np.min(all_values) * 0.95
                max_val = np.max(all_values) * 1.05
                
            fig, ax = self._create_base_plot(min_val, max_val)
            
            scatter = ax.scatter(
                ref_values,
                pred_values,
                color=self.specific_color,
                marker="o",
                s=self.mark_size,
                edgecolors="black",
                linewidths=self.linewidths,
            )
            
            mae = np.mean(np.abs(ref_values - pred_values))
            rmse = np.sqrt(np.mean((ref_values - pred_values)**2))
            max_error = np.max(np.abs(ref_values - pred_values))
            
            display_name = self._display_mlip_name(mlip_name)
            mae_label = f"MAE-{display_name}: {mae:.3f}"
            
            ax.text(
                x=0.05, y=0.95,
                s=mae_label,
                transform=ax.transAxes,
                fontsize=30,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round", alpha=0.5, facecolor="white", 
                    edgecolor="black", pad=0.5
                ),
            )
            
            ax.set_xlabel(f"DFT ({self.task_info['unit']})", fontsize=40)
            ax.set_ylabel(f"{display_name} ({self.task_info['unit']})", fontsize=40)
            ax.tick_params(axis="both", which="major", labelsize=20)
            
            plt.tight_layout()
            
            plot_filename = f"{display_name}_{self.task_type}_parity.png"
            plt.savefig(os.path.join(plot_save_path, plot_filename), dpi=self.dpi)
            plt.close()
            
            print(f"    Parity plot saved: {plot_filename}")
            
            all_metrics[mlip_name] = {
                "MAE": mae, "RMSE": rmse, "Max_Error": max_error, "Num_surface": len(ref_values)
            }
        
        return all_metrics

    def _generate_excel_report(self):
        """Generate Excel report with MLIP-specific sheets and summary."""
        print("Generating Excel report...")
        
        summary_data = []
        all_mlip_data = {}
        
        for mlip_name in self.mlip_list:
            if self.task_type == "surface":
                result_file = f"{mlip_name}_surface_energy_result.json"
            elif self.task_type == "bulk_formation":
                result_file = f"{mlip_name}_bulk_formation_result.json"
            else:  # custom
                result_file = f"{mlip_name}_custom_result.json"
                
            result_path = os.path.join(self.task_result_path, mlip_name, result_file)
            
            if not os.path.exists(result_path):
                print(f"  Warning: Result file not found for {mlip_name}")
                continue
                
            with open(result_path, 'r') as f:
                results = json.load(f)
            
            system_data = []
            ref_values = []
            pred_values = []
            
            for system_name, system_result in results['results'].items():                    
                try:
                    ref_value = system_result["reference"][self.task_info["ref_key"]]
                    pred_value = system_result["mlip_calculation"][self.task_info["pred_key"]]
                    
                    if self.task_type == "surface":
                        ref_value *= 16.022
                        pred_value *= 16.022
                    
                    difference = pred_value - ref_value
                    
                    system_data.append({
                        "System_Name": system_name,
                        "Reference": ref_value,
                        "MLIP_Predicted": pred_value,
                        "Difference": difference
                    })
                    
                    ref_values.append(ref_value)
                    pred_values.append(pred_value)
                    
                except KeyError as e:
                    print(f"  Warning: Missing key {e} for system {system_name} in {mlip_name}")
                    continue
            
            if not system_data:
                print(f"  Warning: No valid data found for {mlip_name}")
                continue
            
            ref_values = np.array(ref_values)
            pred_values = np.array(pred_values)
            mae = np.mean(np.abs(ref_values - pred_values))
            rmse = np.sqrt(np.mean((ref_values - pred_values)**2))
            max_error = np.max(np.abs(ref_values - pred_values))
            num_surface = len(ref_values)
            
            summary_data.append({
                "MLIP_Name": self._display_mlip_name(mlip_name),
                "MAE": mae,
                "RMSE": rmse,
                "Max_Error": max_error,
                "Num_surface": num_surface
            })
            
            all_mlip_data[mlip_name] = system_data
        
        if not summary_data:
            print("Warning: No valid data found for Excel report")
            return
        
        output_file = f"{self.benchmark}_Relative_{self.task_info['name'].replace(' ', '_')}_Analysis.xlsx"
        
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            self._create_summary_sheet_multi(writer, summary_data)
            
            for mlip_name, system_data in all_mlip_data.items():
                self._create_detailed_sheet(writer, system_data, mlip_name)
                
        print(f"Excel file '{output_file}' created successfully.")

    def _create_summary_sheet_multi(self, writer, summary_data):
        """Create MLIP_Data summary sheet for multiple MLIPs."""
        workbook = writer.book
        worksheet = workbook.add_worksheet("MLIP_Data")
        
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
        number_format_3f = workbook.add_format(
            {"num_format": "0.000", "align": "center", "valign": "vcenter"}
        )
        number_format_0f = workbook.add_format(
            {"num_format": "#,##0", "align": "center", "valign": "vcenter"}
        )
        
        headers = ["MLIP Name", f"MAE ({self.task_info['unit']})", f"RMSE ({self.task_info['unit']})", 
                  f"Max Error ({self.task_info['unit']})", "Num_surface"]
        
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        for row, data in enumerate(summary_data, 1):
            worksheet.write(row, 0, data["MLIP_Name"], bold_center_align)
            worksheet.write(row, 1, data["MAE"], number_format_3f)
            worksheet.write(row, 2, data["RMSE"], number_format_3f)
            worksheet.write(row, 3, data["Max_Error"], number_format_3f)
            worksheet.write(row, 4, data["Num_surface"], number_format_0f)
        
        column_widths = [25, 20, 20, 20, 15]
        for col, width in enumerate(column_widths):
            worksheet.set_column(col, col, width)
        
        for row in range(len(summary_data) + 1):
            worksheet.set_row(row, 25)

    def _create_detailed_sheet(self, writer, system_data, mlip_name):
        """Create detailed MLIP sheet."""
        workbook = writer.book
        display_name = self._display_mlip_name(mlip_name)
        worksheet = workbook.add_worksheet(display_name)
        
        header_format = workbook.add_format({
            "align": "center", 
            "valign": "vcenter",
            "text_wrap": True,
            "bold": True
        })
        center_align = workbook.add_format({"align": "center", "valign": "vcenter"})
        number_format_3f = workbook.add_format(
            {"num_format": "0.000", "align": "center", "valign": "vcenter"}
        )
        
        headers = ["System Name", f"Reference ({self.task_info['unit']})", 
                  f"MLIP Predicted ({self.task_info['unit']})", f"Difference ({self.task_info['unit']})"]
        
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        for row, data in enumerate(system_data, 1):
            worksheet.write(row, 0, data["System_Name"], center_align)
            worksheet.write(row, 1, data["Reference"], number_format_3f)
            worksheet.write(row, 2, data["MLIP_Predicted"], number_format_3f)
            worksheet.write(row, 3, data["Difference"], number_format_3f)
        
        column_widths = [30, 25, 25, 25]
        for col, width in enumerate(column_widths):
            worksheet.set_column(col, col, width)
        
        for row in range(len(system_data) + 1):
            worksheet.set_row(row, 25)

    def analysis(self):
        """
        Perform comprehensive relative energy analysis.
        
        Generates parity plots and Excel reports for relative energy benchmarking results.
        """
        print(f"\nRelative Energy Analysis Settings:")
        print(f"   Task: {self.task_info['name']}")
        print(f"   Unit: {self.task_info['unit']}")
        print(f"   Benchmark: {self.benchmark}")
        print(f"   Input Path: {self.calculating_path}")
        print(f"   Output Path: {self.plot_path}")
        print()
        
        print("Starting relative energy analysis...")
        
        _ = self._generate_parity_plots()
        self._generate_excel_report()
        
        print("\nRelative energy analysis completed successfully!")
        return