"""
I/O and path management utilities for CatBench.

This module consolidates file I/O, path generation, and result management functions.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional
from catbench.utils.calculation_utils import NumpyEncoder


# =============================================================================
# PATH GENERATION FUNCTIONS
# =============================================================================

def get_result_directory(mlip_name: str, mode_suffix: str = "") -> str:
    """Get the result directory path for a given MLIP."""
    base_path = os.getcwd()
    if mode_suffix:
        return os.path.join(base_path, f"result_{mode_suffix}", mlip_name)
    return os.path.join(base_path, "result", mlip_name)


def get_raw_data_path(benchmark_name: str) -> str:
    """Get the path to raw data JSON file for a benchmark."""
    return os.path.join(os.getcwd(), f"raw_data/{benchmark_name}_adsorption.json")


def get_raw_data_directory() -> str:
    """Get the raw data directory path."""
    return os.path.join(os.getcwd(), "raw_data")


# =============================================================================
# DIRECTORY CREATION
# =============================================================================

def create_calculation_directories(save_directory: str) -> None:
    """
    Create standard directory structure for calculation results.
    This is actually useful as it creates a complex nested structure.
    """
    subdirs = [
        save_directory,
        os.path.join(save_directory, "traj"),
        os.path.join(save_directory, "log"),
        os.path.join(save_directory, "gases"),
        os.path.join(save_directory, "gases", "POSCARs"),
        os.path.join(save_directory, "gases", "CONTCARs"),
        os.path.join(save_directory, "gases", "traj"),
        os.path.join(save_directory, "gases", "log")
    ]
    for path in subdirs:
        os.makedirs(path, exist_ok=True)


# =============================================================================
# JSON I/O WITH NUMPY SUPPORT
# =============================================================================

def save_json(data: Dict[str, Any], filepath: str, use_numpy_encoder: bool = True) -> None:
    """
    Save data to JSON file with NumpyEncoder support.
    
    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
        use_numpy_encoder: Whether to use NumpyEncoder for numpy arrays
    """
    with open(filepath, "w") as f:
        if use_numpy_encoder:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
        else:
            json.dump(data, f, indent=4)


def load_json(filepath: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load JSON file, returning default if file doesn't exist.
    
    Args:
        filepath: Path to JSON file
        default: Default value if file doesn't exist
        
    Returns:
        Dictionary loaded from JSON or default value
    """
    if not os.path.exists(filepath):
        return default if default is not None else {}
    
    with open(filepath, "r") as f:
        return json.load(f)


# =============================================================================
# RESULT FILE MANAGEMENT
# =============================================================================

def load_existing_results(save_directory: str, mlip_name: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load existing calculation results for restart functionality.
    
    Returns:
        Tuple of (final_result, gas_energies, gas_energies_single)
    """
    # Load main results
    result_path = os.path.join(save_directory, f"{mlip_name}_result.json")
    result_data = load_json(result_path, {})
    
    # Extract completed reactions (excluding calculation_settings)
    final_result = {k: v for k, v in result_data.items() if k != "calculation_settings"}
    
    # Load gas energies
    gas_path = os.path.join(save_directory, f"{mlip_name}_gases.json")
    gas_energies = load_json(gas_path, {})
    
    # Load single point gas energies
    gas_single_path = os.path.join(save_directory, f"{mlip_name}_gases_single_point.json")
    gas_energies_single = load_json(gas_single_path, {})
    
    # Print status if any results were found
    if final_result or gas_energies or gas_energies_single:
        print(f"Found existing results: {len(final_result)} completed reactions, "
              f"{len(gas_energies)} gas energies, "
              f"{len(gas_energies_single)} single point gas energies")
    
    return final_result, gas_energies, gas_energies_single


def save_calculation_results(save_directory: str, mlip_name: str,
                            result_data: Dict[str, Any],
                            gas_energies: Optional[Dict] = None,
                            gas_energies_single: Optional[Dict] = None,
                            calculation_settings: Optional[Dict] = None) -> None:
    """Save calculation results to JSON files."""
    # Add calculation settings if provided
    result_with_settings = result_data.copy()
    if calculation_settings:
        result_with_settings["calculation_settings"] = calculation_settings
    
    # Save main results
    result_path = os.path.join(save_directory, f"{mlip_name}_result.json")
    save_json(result_with_settings, result_path)
    
    # Save gas energies if provided
    if gas_energies is not None:
        gas_path = os.path.join(save_directory, f"{mlip_name}_gases.json")
        save_json(gas_energies, gas_path)
    
    # Save single point gas energies if provided
    if gas_energies_single is not None:
        gas_single_path = os.path.join(save_directory, f"{mlip_name}_gases_single_point.json")
        save_json(gas_energies_single, gas_single_path)


def save_anomaly_detection_results(calculating_path: str, mlip_name: str,
                                  anomaly_data: Dict[str, Any]) -> None:
    """Save anomaly detection results to JSON file."""
    anomaly_path = os.path.join(calculating_path, mlip_name, f"{mlip_name}_anomaly_detection.json")
    save_json(anomaly_data, anomaly_path)


def get_calculation_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract calculation settings for saving to result file."""
    settings_keys = ["optimizer", "f_crit_relax", "n_crit_relax", "rate", "damping", "save_step", "chemical_bond_cutoff"]
    return {k: config.get(k) for k in settings_keys if k in config}