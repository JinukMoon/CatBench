"""
Default configuration settings for CatBench.

This module contains all default parameters used throughout the CatBench library.
These defaults are applied when kwargs don't specify values.
"""

import os

# ============================================================================
# CALCULATION DEFAULTS (from calculation/core.py)
# ============================================================================

CALCULATION_DEFAULTS = {
    # Optimizer settings
    "optimizer": "LBFGS",
    "save_step": 50,  # Save results every N calculations
    
    # Optimization parameters
    "f_crit_relax": 0.05,
    "n_crit_relax": 999,
    "rate": 0.5,
    "damping": 1.0,
    
    # Bond analysis
    "chemical_bond_cutoff": 6.0,  # Chemical bond cutoff in Å for max_bond_change calculation
}

# ============================================================================
# ANALYSIS DEFAULTS (from analysis/catbench_analysis.py)
# ============================================================================

ANALYSIS_DEFAULTS = {
    # Path settings
    "calculating_path": lambda: os.path.join(os.getcwd(), "result"),
    "mlip_list": None,
    "target_adsorbates": None,
    "exclude_adsorbates": None,
    "benchmarking_name": lambda: os.path.basename(os.getcwd()),
    "time_unit": "ms",  # "s", "ms", or "µs"
    
    # Anomaly detection thresholds
    "disp_thrs": 0.5,
    "energy_thrs": 2.0,
    "energy_cutoff": None,  # Energy cutoff for analysis inclusion (None = no filtering)
    "reproduction_thrs": 0.2,
    
    # Adsorbate migration detection thresholds
    "bond_length_change_threshold": 0.2,  # 20% bond length change threshold for anomaly detection
    
    # Plot settings
    "figsize": (9, 8),
    "mark_size": 100,
    "linewidths": 1.5,
    "dpi": 300,
    "legend_off": False,
    "mae_text_off": False,
    "error_bar_display": False,
    "font_setting": False,
    "specific_color": "#2077B5",
    "min": None,
    "max": None,
    
    # Axis label toggles
    "xlabel_off": False,
    "ylabel_off": False,
    
    # Tick control (None = auto)
    "tick_bins": None,
    
    # Tick label decimal places (None = auto)
    "tick_decimal_places": None,
    
    # Font sizes
    "legend_fontsize": 25,
    "tick_labelsize": 25,
    "xlabel_fontsize": 40,
    "ylabel_fontsize": 40,
    "mae_text_fontsize": 30,
    "comparison_legend_fontsize": 15,
    "threshold_xlabel_fontsize": 25,
    "threshold_ylabel_fontsize": 25,
    
    # Grid toggle
    "grid": False,
    
    # Display name mapping for MLIP names (for plots/Excel display only)
    "mlip_name_map": {},
    
    # Plot toggle
    "plot_enabled": True,
}

# Color and marker settings for multi plots
PLOT_COLORS = [
    "blue", "red", "green", "purple", "orange", "brown", "pink", "gray", 
    "olive", "cyan", "magenta", "lime", "indigo", "gold", "darkred", "teal",
    "coral", "turquoise", "salmon", "navy", "maroon", "forestgreen",
    "darkorange", "aqua", "lavender", "khaki", "crimson", "chocolate"
] * 100

PLOT_MARKERS = [
    "o", "^", "s", "p", "*", "h", "D", "H", "d", "<", ">", "v", "8", "P", "X"
] * 100

# ============================================================================
# RELATIVE ENERGY ANALYSIS DEFAULTS (from analysis/relative_analysis.py)
# ============================================================================

RELATIVE_ANALYSIS_DEFAULTS = {
    # Path settings
    "calculating_path": lambda: os.path.join(os.getcwd(), "result"),
    "plot_path": lambda: os.path.join(os.getcwd(), "plot"),
    "benchmark": lambda: os.path.basename(os.getcwd()),
    "task_type": None,
    "mlip_list": None,
    
    # Plot settings (same as catbench_analysis.py)
    "figsize": (9, 8),
    "mark_size": 100,
    "linewidths": 1.5,
    "dpi": 300,
    "font_setting": False,
    "specific_color": "#2077B5",
    "min": None,
    "max": None,
    "grid": False,
}

# ============================================================================
# HELPER FUNCTION TO GET DEFAULTS
# ============================================================================

def get_default(key, defaults_dict):
    """
    Get default value for a given key.
    If the value is callable (lambda), execute it to get the actual value.
    """
    value = defaults_dict.get(key)
    if callable(value):
        return value()
    return value