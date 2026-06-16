"""Utility functions for CatBench analysis."""

import os
import json
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def find_adsorbate(data):
    """Find adsorbate name from reference data."""
    for key in data:
        if key.endswith("star_tot_eng") and key != "star_tot_eng":
            return key[: -len("star_tot_eng")]


def classify_reaction(seed, unphysical, migration, energy):
    """
    Classify a reaction into one of the 5 anomaly categories.

    This is the single shared classifier used by both the live anomaly
    detection (headline Excel counts) and the threshold-sensitivity analysis,
    so the two paths can never disagree at edge cases.

    The priority order is fixed (highest first):
        reproduction_failure > unphysical_relaxation
        > adsorbate_migration > energy_anomaly > normal

    Args:
        seed: bool, any reproduction/seed-range anomaly is present
        unphysical: bool, any convergence or displacement anomaly is present
        migration: bool, adsorbate migration (bond change) anomaly is present
        energy: bool, energy anomaly is present

    Returns:
        str: one of "reproduction_failure", "unphysical_relaxation",
             "adsorbate_migration", "energy_anomaly", "normal"
    """
    if seed:
        return "reproduction_failure"
    elif unphysical:
        return "unphysical_relaxation"
    elif migration:
        return "adsorbate_migration"
    elif energy:
        return "energy_anomaly"
    else:
        return "normal"


def safe_mae(dft_values, mlip_values):
    """
    Compute mean absolute error, returning np.nan (not 0.0) for empty input.

    Reserving np.nan for the no-data case keeps a genuine computed MAE of 0.0
    distinguishable from "no data was available".

    Args:
        dft_values: array-like of reference values
        mlip_values: array-like of predicted values

    Returns:
        float: mean absolute error, or np.nan if there are no data points
    """
    dft_arr = np.asarray(dft_values, dtype=float)
    mlip_arr = np.asarray(mlip_values, dtype=float)
    if dft_arr.size == 0:
        return np.nan
    return float(np.mean(np.abs(dft_arr - mlip_arr)))


def write_cell(worksheet, row, col, value, cell_format):
    """
    Write a numeric cell, rendering nan/None as a blank cell.

    xlsxwriter raises on nan by default; more importantly a blank cell avoids
    presenting "0.000" for a metric that has no data (which would be
    indistinguishable from a genuine zero / perfect score).

    Args:
        worksheet: xlsxwriter worksheet
        row: row index
        col: column index
        value: numeric value (may be nan/None for no-data)
        cell_format: xlsxwriter format object
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        worksheet.write(row, col, "", cell_format)
    else:
        worksheet.write(row, col, value, cell_format)


def min_max(dft_values):
    """
    Calculate plot axis limits with padding.
    
    Args:
        dft_values: Array-like of numerical values
        
    Returns:
        tuple: (min_val, max_val) with 10% padding on each end
    """
    min_value = float(np.min(dft_values))
    max_value = float(np.max(dft_values))
    range_value = max_value - min_value
    min_val = min_value - 0.1 * range_value
    max_val = max_value + 0.1 * range_value
    return min_val, max_val


def set_matplotlib_font(font_path, font_family):
    """Set matplotlib font configuration."""
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    font_name = prop.get_name()
    plt.rcParams['font.family'] = font_family
    if font_family == 'sans-serif':
        plt.rcParams['font.sans-serif'] = [font_name]
    elif font_family == 'serif':
        plt.rcParams['font.serif'] = [font_name]
    elif font_family == 'monospace':
        plt.rcParams['font.monospace'] = [font_name]


def get_calculator_keys(reaction_data):
    """
    Extract calculator keys from reaction data efficiently.
    
    Args:
        reaction_data: Dictionary containing reaction data
        
    Returns:
        list: List of calculator keys (string format)
    """
    return [key for key in reaction_data if isinstance(key, (int, str)) and str(key).isdigit()]


def get_median_calculator_key(reaction_data):
    """
    Get median calculator key from reaction results.
    
    Args:
        reaction_data: Dictionary containing reaction data with 'final' section
        
    Returns:
        str: Median calculator key
    """
    return str(reaction_data["final"]["median_num"])


def get_ads_eng_range(data_dict):
    """
    Extract adsorption energy range from calculation results.
    
    Args:
        data_dict: Dictionary containing calculation results with ads_eng values
        
    Returns:
        tuple: (min_energy, max_energy) from all reactions
    """
    ads_eng_values = []
    for key in data_dict:
        if isinstance(key, (int, str)) and str(key).isdigit():
            ads_eng_values.append(data_dict[key]["ads_eng"])
    return min(ads_eng_values), max(ads_eng_values)


def prepare_plot_data(ads_data, types, adsorbate=None):
    """
    Extract and prepare data for plotting from analysis results.
    
    Args:
        ads_data: Dictionary containing adsorption analysis results
        types: List of data types to include (e.g., ['normal', 'anomaly'])
        adsorbate (optional): Specific adsorbate to filter by
        
    Returns:
        tuple: (dft_values, mlip_values) arrays ready for plotting
    """
    if adsorbate:
        data_source = ads_data[adsorbate]
    else:
        data_source = ads_data["all"]
    
    # Handle different data structures (5-category vs single calculation)
    if "normal" in data_source:
        dft_values = []
        mlip_values = []
        mlip_mins = []
        mlip_maxs = []
        
        # Define anomaly categories (everything except normal)
        anomaly_categories = ["energy_anomaly", "adsorbate_migration", "unphysical_relaxation", "reproduction_failure"]
        
        for type_name in types:
            if type_name == "anomaly":
                # Combine all anomaly categories
                for anomaly_type in anomaly_categories:
                    if anomaly_type in data_source:
                        dft_values.append(data_source[anomaly_type]["DFT"])
                        mlip_values.append(data_source[anomaly_type]["MLIP"])
                        if "MLIP_min" in data_source[anomaly_type]:
                            mlip_mins.append(data_source[anomaly_type]["MLIP_min"])
                            mlip_maxs.append(data_source[anomaly_type]["MLIP_max"])
            elif type_name in data_source:
                # Single specific category
                dft_values.append(data_source[type_name]["DFT"])
                mlip_values.append(data_source[type_name]["MLIP"])
                if "MLIP_min" in data_source[type_name]:
                    mlip_mins.append(data_source[type_name]["MLIP_min"])
                    mlip_maxs.append(data_source[type_name]["MLIP_max"])
        
        # Handle empty arrays
        if dft_values:
            dft_values = np.concatenate(dft_values)
            mlip_values = np.concatenate(mlip_values)
        else:
            dft_values = np.array([])
            mlip_values = np.array([])
        
        result = {"DFT": dft_values, "MLIP": mlip_values}
        
        if mlip_mins:
            mlip_mins = np.concatenate(mlip_mins)
            mlip_maxs = np.concatenate(mlip_maxs)
            result["MLIP_min"] = mlip_mins
            result["MLIP_max"] = mlip_maxs
            
        return result
    else:
        # Single calculation case
        return {
            "DFT": data_source["all"]["DFT"],
            "MLIP": data_source["all"]["MLIP"]
        } 