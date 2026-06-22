"""
Adsorption energy calculation module for CatBench.

This module provides the AdsorptionCalculation class for performing
adsorption energy benchmarking calculations with MLIPs.
"""

import json
import os
import shutil
import traceback
import warnings

import numpy as np
from ase.geometry import find_mic
from ase.io import write

from catbench.config import CALCULATION_DEFAULTS, get_default
from catbench.utils.calculation_utils import (
    energy_cal_gas, energy_cal_single, energy_cal,
    calc_displacement, find_median_index, get_fixed_indices, NumpyEncoder
)
from catbench.utils.io_utils import (
    create_calculation_directories, get_result_directory, get_raw_data_path,
    load_existing_results, save_calculation_results, get_calculation_settings
)
from catbench.utils.structure_dedup import reuse_key


class AdsorptionCalculation:
    """
    Adsorption energy calculation class for MLIP benchmarking.
    
    This class provides a comprehensive interface for running adsorption energy
    benchmarks with Machine Learning Interatomic Potentials (MLIPs) across 
    different calculation modes and datasets.
    
    Calculation Modes:
        basic: Standard benchmarking with multiple calculator seeds
               - Runs full relaxation calculations for both slab and adsorbate
               - Includes single-point calculations for comparison
               - Collects data for later anomaly detection during analysis
               - Equivalent to core.execute_benchmark function
               
        oc20: Open Catalyst 2020 dataset specific calculations
              - For OC20 data structure and workflow
              - Focuses on adsorbate calculations only
              - Includes single-point calculations for comparison
              - Equivalent to core.execute_benchmark_OC20 function
    
    Input File Requirements:
        - JSON files: {benchmark}_adsorption.json containing reference structures and energies
        - Calculator objects: ASE-compatible MLIP calculators (list)
        
    Output Files:
        - JSON result files: {MLIP_name}_result.json with detailed calculation data
        - JSON gas files: {MLIP_name}_gases.json with gas molecule energies
        - Trajectory files: ASE trajectory files in extxyz format
        - Log files: Detailed calculation logs for each structure
        
    Args:
        calculators (list): List of ASE-compatible MLIP calculators for benchmarking.
        mode (str): Calculation mode - "basic" or "oc20". Default: "basic"
        mlip_name (str): Name identifier for the MLIP being benchmarked.
        benchmark (str): Name of the benchmark dataset (JSON file basename).
        f_crit_relax (float, optional): Force convergence criterion in eV/Å. Default: 0.05
        n_crit_relax (int, optional): Maximum optimization steps. Default: 999
        rate (float, optional): If set, fixes the bottom `rate` fraction of atoms by z-coordinate (legacy override). Default None = use the structure's stored FixAtoms constraints.
        damping (float, optional): Damping factor for optimization. Default: 1.0
        optimizer (str, optional): ASE optimizer name. Default: "LBFGS"
        save_step (int, optional): Save results every N calculations. Default: 50
        
    Raises:
        ValueError: If mode is not valid or required parameters are missing.
        FileNotFoundError: If benchmark data file is not found.
    """
    
    def __init__(self, calculators, mode="basic", **kwargs):
        """
        Initialize adsorption energy calculation.
        
        Args:
            calculators: List of ASE calculators for MLIP evaluation
            mode (str): Calculation mode ("basic" or "oc20"). Default: "basic"
            **kwargs: Additional configuration parameters including:
                - benchmark: Name of the benchmark dataset  
                - save_step: Save results every N calculations
                - restart: Continue from previous calculation
                - f_crit_relax: Force convergence criterion for relaxation
                - n_crit_relax: Maximum number of relaxation steps
                - rate: If set (float), fix bottom `rate` fraction of atoms by z-coordinate (legacy override); None = use the structure's stored FixAtoms constraints
                - damping: Damping factor for optimization
                
        Raises:
            ValueError: If mode is invalid or calculators list is empty
        """
        # Validate mode
        valid_modes = ["basic", "oc20"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got: {mode}")
        
        self.mode = mode
        
        # Handle calculator input - both modes require list of calculators
        if not isinstance(calculators, list) or len(calculators) == 0:
            raise ValueError("Both basic and OC20 modes require a non-empty list of calculators")
        self.calculators = calculators
        
        # Validate required parameters
        required_keys = ["mlip_name", "benchmark"]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required parameter: {key}")
        
        # Store configuration
        self.config = kwargs
        self.mlip_name = kwargs["mlip_name"]
        self.benchmark = kwargs["benchmark"]
        
        # Set configuration parameters
        self._set_default_params()
        
    def _set_default_params(self):
        """Set configuration parameters with appropriate defaults."""
        # Warn (don't silently ignore) on knobs removed in 1.1.1: fixing is now
        # purely the structure's stored FixAtoms, so an explicit rate/z_target/
        # fix_z would change a user's intended behavior without notice.
        for legacy in ("rate", "z_target", "fix_z"):
            if legacy in self.config:
                warnings.warn(
                    f"'{legacy}' was removed in CatBench 1.1.1 and is ignored; "
                    "atom fixing now uses the structure's stored FixAtoms only.",
                    stacklevel=2,
                )
        # Apply configuration defaults
        for key, default_value in CALCULATION_DEFAULTS.items():
            if key not in self.config:
                self.config[key] = get_default(key, CALCULATION_DEFAULTS)

    def _relax_sig(self):
        """Signature of the settings that determine a relaxation result. The slab
        cache is only valid while these are unchanged; a change invalidates it."""
        return [
            str(self.config.get("optimizer")),
            self.config.get("f_crit_relax"),
            self.config.get("n_crit_relax"),
            self.config.get("damping"),
        ]
    
    def run(self):
        """
        Run the benchmarking calculation based on the specified mode.
        
        Returns:
            str: Path to the results directory
        """
        print(f"Starting {self.mlip_name} Benchmarking in {self.mode} mode")
        
        if self.mode == "basic":
            return self._run_basic()
        else:  # oc20
            return self._run_oc20()
    
    def _load_data(self):
        """Load benchmark data from JSON file."""
        path_json = get_raw_data_path(self.benchmark)
        if not os.path.exists(path_json):
            raise FileNotFoundError(
                f"Data file not found: {path_json}\n"
                f"Please run preprocessing to generate the JSON file."
            )
        
        from catbench.utils.data_utils import load_catbench_json
        return load_catbench_json(path_json)
    
    def _setup_directories(self, mode_suffix=""):
        """Setup output directories."""
        save_directory = get_result_directory(self.mlip_name, mode_suffix=mode_suffix)
        # Always create base directory for result.json
        os.makedirs(save_directory, exist_ok=True)
        # Only create subdirectories if save_files is True
        if self.config.get("save_files", True):
            create_calculation_directories(save_directory)
        return save_directory
    
    def _load_existing_results(self, save_directory):
        """Load existing results if available (automatic restart)."""
        try:
            final_result, gas_energies, gas_energies_single = load_existing_results(
                save_directory, self.mlip_name
            )
        except FileNotFoundError:
            print("Beginning calculation from scratch.")
            final_result, gas_energies, gas_energies_single = {}, {}, {}

        # Slab-relaxation cache (1.1.1) — restart-safe sibling file, loaded
        # independently of the result file so an interrupted run resumes without
        # re-relaxing already-seen clean slabs.
        slab_energies = {}
        slab_path = os.path.join(save_directory, f"{self.mlip_name}_slab_energies.json")
        if os.path.exists(slab_path):
            try:
                with open(slab_path) as f:
                    slab_energies = json.load(f)
            except Exception:
                slab_energies = {}
            # Cached relaxations are valid only for the settings that produced them.
            # If the run's relaxation settings changed, discard the cache so
            # structures are re-relaxed instead of silently reusing stale results.
            if slab_energies.get("__relax_config__") != self._relax_sig():
                if slab_energies:
                    print("Relaxation settings changed since the slab cache was "
                          "written; discarding cache and recomputing.")
                slab_energies = {}
        return final_result, gas_energies, gas_energies_single, slab_energies
    
    def _run_basic(self):
        """Run basic benchmarking mode (full features with multiple calculators)."""
        ref_data = self._load_data()
        save_directory = self._setup_directories()
        final_result, gas_energies, gas_energies_single, slab_energies = self._load_existing_results(save_directory)
        failed = {}

        print("Starting calculations...")
        for index, key in enumerate(ref_data):
            # Skip if already calculated
            if key in final_result:
                print(f"Skipping already calculated {key}")
                continue

            # Clean up any incomplete attempts (only if save_files is True)
            if self.config.get("save_files", True):
                log_path = f"{save_directory}/log/{key}"
                traj_path = f"{save_directory}/traj/{key}"
                if os.path.exists(log_path):
                    shutil.rmtree(log_path)
                    print(f"Removed existing log directory for {key}")
                if os.path.exists(traj_path):
                    shutil.rmtree(traj_path)
                    print(f"Removed existing trajectory directory for {key}")
            
            try:
                print(f"[{index+1}/{len(ref_data)}] {key}")
                result = self._process_reaction_basic(key, ref_data[key], save_directory, gas_energies, gas_energies_single, slab_energies)
                final_result[key] = result["reaction_result"]

                # Save results every save_step calculations
                if len(final_result) % self.config["save_step"] == 0:
                    print(f"Saving results at {len(final_result)} calculations...")
                    self._save_results_basic(save_directory, final_result, gas_energies, gas_energies_single, slab_energies, failed)

            except Exception as e:
                print(f"Error occurred while processing {key}: {str(e)}")
                print("Skipping to next reaction...")
                failed[key] = {"error": repr(e), "traceback": traceback.format_exc()[-2000:]}
                continue

        # Final save to ensure all results are saved
        print(f"Final save: {len(final_result)} total calculations")
        self._save_results_basic(save_directory, final_result, gas_energies, gas_energies_single, slab_energies, failed)

        print(f"{len(final_result)} succeeded, {len(failed)} failed")
        if failed:
            print(f"Failed reactions: {list(failed.keys())}")
        print(f"{self.mlip_name} Benchmarking Finish")
        return save_directory
    
    def _process_reaction_basic(self, key, reaction_data, save_directory, gas_energies, gas_energies_single, slab_energies):
        """
        Process a single adsorption reaction in basic mode.
        
        Args:
            key: Unique reaction identifier
            reaction_data: Dictionary containing reaction information
            save_directory: Path to save intermediate results
            gas_energies: Pre-computed gas phase energies
            gas_energies_single: Single point gas energies
            
        Returns:
            dict: Processing result containing reaction outcomes and calculations
        """
        # Get adsorbate indices from data - moved here for early access
        if "adsorbate_indices" not in reaction_data:
            raise KeyError(f"Missing 'adsorbate_indices' key in data for reaction {key}. "
                          "Please re-run preprocessing to generate adsorbate indices.")
        adsorbate_indices = reaction_data["adsorbate_indices"]
        
        # Initialize result structure
        result = {
            "reference": {
                "ads_eng": reaction_data["ref_ads_eng"]
            }
        }
        
        # Add reference energies
        for structure in reaction_data["raw"]:
            if "gas" not in str(structure):
                result["reference"][f"{structure}_tot_eng"] = reaction_data["raw"][structure]["energy_ref"]
        
        # Add adsorbate indices right after reference
        result["adsorbate_indices"] = adsorbate_indices
        
        # Add single-point calculation using first calculator
        ads_energy_single = 0
        slab_energy_single = None
        adslab_energy_single = None
        
        for structure in reaction_data["raw"]:
            if "gas" not in str(structure):
                POSCAR_str = reaction_data["raw"][structure]["atoms"]
                if structure == "star":
                    # Single-point clean-slab energy is frame-invariant -> cache &
                    # reuse across frame-equivalent slabs (key tagged '|sp', calc[0]).
                    sp_key = f"{reuse_key(POSCAR_str, reaction_data['raw'][structure].get('energy_ref'))}|sp"
                    if self.config.get("slab_cache", True) and sp_key in slab_energies:
                        energy_calculated = slab_energies[sp_key]
                    else:
                        energy_calculated = energy_cal_single(self.calculators[0], POSCAR_str)
                        if self.config.get("slab_cache", True):
                            slab_energies[sp_key] = energy_calculated
                    slab_energy_single = energy_calculated
                else:
                    energy_calculated = energy_cal_single(self.calculators[0], POSCAR_str)
                    adslab_energy_single = energy_calculated
                ads_energy_single += energy_calculated * reaction_data["raw"][structure]["stoi"]
            else:  # Gas molecule - use single point
                gas_tag = structure  # Simplified: no suffix for single point
                if gas_tag in gas_energies_single:
                    ads_energy_single += gas_energies_single[gas_tag] * reaction_data["raw"][structure]["stoi"]
                else:
                    print(f"{gas_tag} single point calculating")
                    gas_atoms = reaction_data["raw"][structure]["atoms"]
                    # Remove calculator if it exists to prevent deepcopy issues
                    if hasattr(gas_atoms, 'calc'):
                        gas_atoms.calc = None
                    # Now use energy_cal_single which will deepcopy safely
                    gas_energy = energy_cal_single(self.calculators[0], gas_atoms)
                    gas_energies_single[gas_tag] = gas_energy
                    ads_energy_single += gas_energy * reaction_data["raw"][structure]["stoi"]
        
        result["single_calculation"] = {
            "ads_eng": ads_energy_single,
            "slab_tot_eng": slab_energy_single,
            "adslab_tot_eng": adslab_energy_single,
        }
        
        # Setup paths (conditionally based on save_files)
        if self.config.get("save_files", True):
            traj_path = f"{save_directory}/traj/{key}"
            log_path = f"{save_directory}/log/{key}"
            os.makedirs(traj_path, exist_ok=True)
            os.makedirs(log_path, exist_ok=True)
        else:
            traj_path = None
            log_path = None
        
        POSCAR_star = reaction_data["raw"]["star"]["atoms"]

        # Initialize tracking arrays
        informs = {
            "ads_eng": [],
            "slab_max_disp": [],
            "slab_pos_mae": [],
            "slab_pos_rmsd": [],
            "adslab_max_disp": [],
            "adslab_pos_mae": [],
            "adslab_pos_rmsd": [],
            "slab_seed": [],
            "ads_seed": []
        }
        
        time_total_slab = 0
        time_total_ads = 0
        time_consumed = 0
        steps_total_slab = 0
        steps_total_ads = 0
        
        # Precompute the seed-independent reuse_key per non-gas structure ONCE.
        # The geometry fingerprint is O(n_atoms^2) and identical across seeds, so
        # computing it inside the seed loop would repeat it (n_seeds - 1)x.
        reuse_keys = {
            s: reuse_key(reaction_data["raw"][s]["atoms"],
                         reaction_data["raw"][s].get("energy_ref"))
            for s in reaction_data["raw"] if "gas" not in str(s)
        }

        # Run calculations for each calculator
        for i in range(len(self.calculators)):
            ads_energy_calc = 0
            
            # Store initial and final structures for analysis
            slab_initial = None
            slab_final = None
            adslab_initial = None
            adslab_final = None
            # Adslab reuse bookkeeping (mirrors the clean-slab cache)
            adslab_cache_key = None
            adslab_cached = False
            cached_max_bond_change = None
            cached_substrate_disp = None

            for structure in reaction_data["raw"]:
                if "gas" not in str(structure) and structure == "star":
                    # Clean slab: relax once per (geometry+fix, seed) and reuse the
                    # result for every frame-equivalent slab (energy + displacement
                    # are frame-invariant -> identical result, pure speedup).
                    POSCAR_str = reaction_data["raw"][structure]["atoms"]
                    fixed_indices = get_fixed_indices(POSCAR_str)
                    slab_key = f"{reuse_keys[structure]}_{i}th"
                    cached = slab_energies.get(slab_key) if self.config.get("slab_cache", True) else None
                    if cached is not None:
                        slab_energy = cached["slab_tot_eng"]
                        slab_steps = cached["slab_steps"]
                        slab_displacement_stats = {
                            "max_disp": cached["slab_max_disp"],
                            "mae_mobile": cached["slab_pos_mae"],
                            "rmsd_mobile": cached["slab_pos_rmsd"],
                        }
                        slab_energy_change = cached["slab_energy_change"]
                        slab_time = 0.0  # cache hit: no relaxation performed
                        slab_initial = POSCAR_str.copy()
                        slab_final = POSCAR_str.copy()  # placeholder (unused downstream)
                    else:
                        (
                            energy_calculated,
                            steps_calculated,
                            CONTCAR_calculated,
                            time_calculated,
                            energy_change,
                        ) = energy_cal(
                            self.calculators[i], POSCAR_str,
                            self.config["f_crit_relax"], self.config["n_crit_relax"],
                            self.config["damping"], fixed_indices, self.config["optimizer"],
                            f"{log_path}/{structure}_{i}.txt" if log_path else None,
                            f"{traj_path}/{structure}_{i}" if traj_path else None,
                        )
                        slab_steps = steps_calculated
                        slab_displacement_stats = calc_displacement(POSCAR_str, CONTCAR_calculated, fixed_indices)
                        slab_energy = energy_calculated
                        slab_time = time_calculated
                        slab_energy_change = energy_change
                        slab_initial = POSCAR_str.copy()
                        slab_final = CONTCAR_calculated.copy()
                        if self.config.get("slab_cache", True):
                            slab_energies[slab_key] = {
                                "slab_tot_eng": slab_energy, "slab_steps": slab_steps,
                                "slab_max_disp": slab_displacement_stats["max_disp"],
                                "slab_pos_mae": slab_displacement_stats["mae_mobile"],
                                "slab_pos_rmsd": slab_displacement_stats["rmsd_mobile"],
                                "slab_energy_change": slab_energy_change,
                            }
                    ads_energy_calc += slab_energy * reaction_data["raw"][structure]["stoi"]
                    time_consumed += slab_time
                    if cached is None:
                        # Only actual relaxations count toward efficiency totals, so
                        # time_per_step stays consistent whether or not the slab was
                        # reused (a cache hit did 0 work: 0 time AND 0 steps).
                        time_total_slab += slab_time
                        steps_total_slab += slab_steps

                elif "gas" not in str(structure):  # adslab
                    # Relax once per (geometry+fix, adsorbate_indices, seed) and reuse
                    # the result for every identical adslab. Same machinery as the clean
                    # slab; the key also pins adsorbate_indices because the adslab metrics
                    # (bond change, substrate displacement) depend on which atoms are the
                    # adsorbate. The relaxed-energy and the frame-invariant metrics are
                    # identical for an identical adslab -> same result, pure speedup.
                    POSCAR_str = reaction_data["raw"][structure]["atoms"]
                    fixed_indices = get_fixed_indices(POSCAR_str)
                    adslab_cache_key = f"ads:{reuse_keys[structure]}|{adsorbate_indices}_{i}th"
                    cached = slab_energies.get(adslab_cache_key) if self.config.get("slab_cache", True) else None
                    if cached is not None:
                        ads_energy = cached["adslab_tot_eng"]
                        ads_step = cached["adslab_steps"]
                        ads_energy_change = cached["adslab_energy_change"]
                        ads_displacement_stats = {
                            "max_disp": cached["adslab_max_disp"],
                            "mae_mobile": cached["adslab_pos_mae"],
                            "rmsd_mobile": cached["adslab_pos_rmsd"],
                        }
                        cached_max_bond_change = cached["max_bond_change"]
                        cached_substrate_disp = cached["substrate_displacement"]
                        ads_time = 0.0  # cache hit: no relaxation performed
                        adslab_cached = True
                        ads_energy_calc += ads_energy * reaction_data["raw"][structure]["stoi"]
                    else:
                        (
                            energy_calculated,
                            steps_calculated,
                            CONTCAR_calculated,
                            time_calculated,
                            energy_change,
                        ) = energy_cal(
                            self.calculators[i], POSCAR_str,
                            self.config["f_crit_relax"], self.config["n_crit_relax"],
                            self.config["damping"], fixed_indices, self.config["optimizer"],
                            f"{log_path}/{structure}_{i}.txt" if log_path else None,
                            f"{traj_path}/{structure}_{i}" if traj_path else None,
                        )
                        ads_energy_calc += energy_calculated * reaction_data["raw"][structure]["stoi"]
                        time_consumed += time_calculated
                        ads_step = steps_calculated
                        ads_displacement_stats = calc_displacement(POSCAR_str, CONTCAR_calculated, fixed_indices)
                        ads_energy = energy_calculated
                        ads_time = time_calculated
                        ads_energy_change = energy_change
                        time_total_ads += time_calculated
                        steps_total_ads += steps_calculated
                        adslab_initial = POSCAR_str.copy()
                        adslab_final = CONTCAR_calculated.copy()

                else:  # Gas molecule
                    gas_tag = f"{structure}_{i}th"
                    if gas_tag in gas_energies:
                        ads_energy_calc += gas_energies[gas_tag] * reaction_data["raw"][structure]["stoi"]
                    else:
                        print(f"{gas_tag} calculating")
                        if self.config.get("save_files", True):
                            gas_CONTCAR, gas_energy = energy_cal_gas(
                                self.calculators[i],
                                reaction_data["raw"][structure]["atoms"],
                                self.config["f_crit_relax"],
                                f"{save_directory}/gases/POSCARs/POSCAR_{gas_tag}",
                                self.config["optimizer"],
                                f"{save_directory}/gases/log/{gas_tag}.txt",
                                f"{save_directory}/gases/traj/{gas_tag}",
                            )
                            write(f"{save_directory}/gases/CONTCARs/CONTCAR_{gas_tag}", gas_CONTCAR)
                        else:
                            gas_CONTCAR, gas_energy = energy_cal_gas(
                                self.calculators[i],
                                reaction_data["raw"][structure]["atoms"],
                                self.config["f_crit_relax"],
                                None,  # No save path
                                self.config["optimizer"],
                                None,  # No log path
                                None,  # No trajectory path
                            )
                        gas_energies[gas_tag] = gas_energy
                        ads_energy_calc += gas_energy * reaction_data["raw"][structure]["stoi"]
            
            # Calculate bond change and substrate displacement
            max_bond_change = 0.0
            substrate_disp = 0.0

            if adslab_cached:
                # Reused adslab: both metrics are frame-invariant functions of
                # (adslab_initial, adslab_final, adsorbate_indices), so the cached
                # values are exactly what a fresh relaxation would reproduce.
                max_bond_change = cached_max_bond_change
                substrate_disp = cached_substrate_disp
            elif adslab_initial is not None and adslab_final is not None and adsorbate_indices:
                # Calculate max bond change for adsorbate
                max_bond_change = self._calculate_max_bond_change(
                    adslab_initial, adslab_final, adsorbate_indices
                )

                # Calculate substrate displacement in adslab
                if slab_initial is not None and slab_final is not None:
                    substrate_disp = self._calculate_substrate_displacement(
                        slab_initial, slab_final, adslab_initial, adslab_final, adsorbate_indices
                    )

                # Store the full adslab result so an identical adslab (same seed) reuses
                # it instead of relaxing again.
                if adslab_cache_key is not None and self.config.get("slab_cache", True):
                    slab_energies[adslab_cache_key] = {
                        "adslab_tot_eng": ads_energy,
                        "adslab_steps": ads_step,
                        "adslab_energy_change": ads_energy_change,
                        "adslab_max_disp": ads_displacement_stats["max_disp"],
                        "adslab_pos_mae": ads_displacement_stats["mae_mobile"],
                        "adslab_pos_rmsd": ads_displacement_stats["rmsd_mobile"],
                        "max_bond_change": max_bond_change,
                        "substrate_displacement": substrate_disp,
                    }
            
            # Store results for this calculator
            result[f"{i}"] = {
                "ads_eng": ads_energy_calc,
                "slab_tot_eng": slab_energy,
                "adslab_tot_eng": ads_energy,
                # Slab displacement metrics
                "slab_max_disp": slab_displacement_stats["max_disp"],
                "slab_pos_mae": slab_displacement_stats["mae_mobile"],
                "slab_pos_rmsd": slab_displacement_stats["rmsd_mobile"],
                # Adslab displacement metrics  
                "adslab_max_disp": ads_displacement_stats["max_disp"],
                "adslab_pos_mae": ads_displacement_stats["mae_mobile"],
                "adslab_pos_rmsd": ads_displacement_stats["rmsd_mobile"],
                # Include: Bond change and substrate displacement
                "max_bond_change": max_bond_change,
                "substrate_displacement": substrate_disp,
                # Energy changes
                "slab_energy_change": slab_energy_change,
                "adslab_energy_change": ads_energy_change,
                # Timing and steps
                "slab_time": slab_time,
                "adslab_time": ads_time,
                "slab_steps": slab_steps,
                "adslab_steps": ads_step,
            }
            
            # Collect data for seed analysis
            informs["ads_eng"].append(ads_energy_calc)
            informs["slab_max_disp"].append(slab_displacement_stats["max_disp"])
            informs["slab_pos_mae"].append(slab_displacement_stats["mae_mobile"])
            informs["slab_pos_rmsd"].append(slab_displacement_stats["rmsd_mobile"])
            informs["adslab_max_disp"].append(ads_displacement_stats["max_disp"])
            informs["adslab_pos_mae"].append(ads_displacement_stats["mae_mobile"])
            informs["adslab_pos_rmsd"].append(ads_displacement_stats["rmsd_mobile"])
            informs["slab_seed"].append(slab_energy)
            informs["ads_seed"].append(ads_energy)
        
        # Analyze seed variations (for analysis stage)
        ads_med_index, ads_med_eng = find_median_index(informs["ads_eng"])
        slab_seed_range = np.max(np.array(informs["slab_seed"])) - np.min(np.array(informs["slab_seed"]))
        ads_seed_range = np.max(np.array(informs["ads_seed"])) - np.min(np.array(informs["ads_seed"]))
        ads_eng_seed_range = np.max(np.array(informs["ads_eng"])) - np.min(np.array(informs["ads_eng"]))
        
        # Anomaly detection performed in analysis phase
        
        # Calculate efficiency metrics
        total_time = time_total_slab + time_total_ads
        total_steps = steps_total_slab + steps_total_ads
        
        # Get adslab atoms count for weighted average
        adslab_key = [key for key in reaction_data["raw"].keys() if "star" in key and key != "star"][0]
        adslab_atoms = len(reaction_data["raw"][adslab_key]["atoms"])
        slab_atoms = len(POSCAR_star)
        
        # Weighted average: total_atom_steps = slab_steps * slab_atoms + ads_steps * ads_atoms
        total_atom_steps = steps_total_slab * slab_atoms + steps_total_ads * adslab_atoms
        step_weighted_atoms = total_atom_steps / total_steps if total_steps > 0 else 0
        
        # Final summary
        result["final"] = {
            "ads_eng_median": ads_med_eng,
            "median_num": ads_med_index,
            "slab_max_disp": np.max(np.array(informs["slab_max_disp"])),
            "adslab_max_disp": np.max(np.array(informs["adslab_max_disp"])),
            "slab_seed_range": slab_seed_range,
            "ads_seed_range": ads_seed_range,
            "ads_eng_seed_range": ads_eng_seed_range,
            "time_total_slab": time_total_slab,
            "time_total_adslab": time_total_ads,
            "steps_total_slab": steps_total_slab,
            "steps_total_adslab": steps_total_ads,
            "step_weighted_atoms": step_weighted_atoms,  # Step-weighted average atom count
            "time_per_step": total_time / total_steps if total_steps > 0 else 0,
            "time_per_step_per_atom": total_time / total_atom_steps if total_atom_steps > 0 else 0,  # Weighted average
        }
        
        return {"reaction_result": result, "time_consumed": time_consumed}
    
    def _run_oc20(self):
        """Run OC20 benchmarking mode (adsorbate-only calculations)."""
        ref_data = self._load_data()
        save_directory = self._setup_directories()
        final_result, _, _, _ = self._load_existing_results(save_directory)  # gas/slab caches unused for OC20
        failed = {}

        print("Starting calculations...")
        for index, key in enumerate(ref_data):
            # Skip if already calculated
            if key in final_result:
                print(f"Skipping already calculated {key}")
                continue

            try:
                print(f"[{index+1}/{len(ref_data)}] {key}")
                result = self._process_reaction_oc20(key, ref_data[key], save_directory)
                final_result[key] = result["reaction_result"]

                # Save results every save_step calculations
                if len(final_result) % self.config["save_step"] == 0:
                    print(f"Saving results at {len(final_result)} calculations...")
                    self._save_results_oc20(save_directory, final_result, failed)

            except Exception as e:
                print(f"Error occurred while processing {key}: {str(e)}")
                print("Skipping to next reaction...")
                failed[key] = {"error": repr(e), "traceback": traceback.format_exc()[-2000:]}
                continue

        # Final save to ensure all results are saved
        print(f"Final save: {len(final_result)} total calculations")
        self._save_results_oc20(save_directory, final_result, failed)

        print(f"{len(final_result)} succeeded, {len(failed)} failed")
        if failed:
            print(f"Failed reactions: {list(failed.keys())}")
        print(f"{self.mlip_name} Benchmarking Finish")
        return save_directory
    
    def _process_reaction_oc20(self, key, reaction_data, save_directory):
        """Process a single reaction in OC20 mode."""
        result = {
            "reference": {
                "ads_eng": reaction_data["ref_ads_eng"]
            }
        }
        
        # Add reference energies (only adslab, no gas)
        for structure in reaction_data["raw"]:
            if "gas" not in str(structure):
                result["reference"][f"{structure}_tot_eng"] = reaction_data["raw"][structure]["energy_ref"]
        
        # Add single-point calculation using first calculator (only adslab)
        ads_energy_single = 0
        
        for structure in reaction_data["raw"]:
            if "gas" not in str(structure) and structure != "star":
                POSCAR_str = reaction_data["raw"][structure]["atoms"]
                energy_calculated = energy_cal_single(self.calculators[0], POSCAR_str)
                ads_energy_single = energy_calculated
        
        result["single_calculation"] = {
            "ads_eng": ads_energy_single,
        }
        
        # Setup paths (conditionally based on save_files)
        if self.config.get("save_files", True):
            traj_path = f"{save_directory}/traj/{key}"
            log_path = f"{save_directory}/log/{key}"
            os.makedirs(traj_path, exist_ok=True)
            os.makedirs(log_path, exist_ok=True)
        else:
            traj_path = None
            log_path = None
        
        POSCAR_star = reaction_data["raw"]["star"]["atoms"]

        # Get adsorbate indices from data
        if "adsorbate_indices" not in reaction_data:
            raise KeyError(f"Missing 'adsorbate_indices' key in data for reaction {key}. "
                          "Please re-run preprocessing to generate adsorbate indices.")
        adsorbate_indices = reaction_data["adsorbate_indices"]
        
        informs = {
            "ads_eng": [],
            "adslab_max_disp": [],
            "adslab_pos_mae": [],
            "adslab_pos_rmsd": []
        }
        
        time_consumed = 0
        time_total_ads = 0
        steps_total_ads = 0
        
        for i in range(len(self.calculators)):
            for structure in reaction_data["raw"]:
                if "gas" not in str(structure) and structure != "star":
                    POSCAR_str = reaction_data["raw"][structure]["atoms"]
                    fixed_indices = get_fixed_indices(POSCAR_str)
                    (
                        ads_energy,
                        steps_calculated,
                        CONTCAR_calculated,
                        time_calculated,
                        ads_energy_change,
                    ) = energy_cal(
                        self.calculators[i],
                        POSCAR_str,
                        self.config["f_crit_relax"],
                        self.config["n_crit_relax"],
                        self.config["damping"],
                        fixed_indices,
                        self.config["optimizer"],
                        f"{log_path}/{structure}_{i}.txt" if log_path else None,
                        f"{traj_path}/{structure}_{i}" if traj_path else None,
                    )
                    time_consumed += time_calculated

                    ads_step = steps_calculated
                    ads_displacement_stats = calc_displacement(POSCAR_str, CONTCAR_calculated, fixed_indices)
                    ads_time = time_calculated
                    time_total_ads += time_calculated
                    steps_total_ads += steps_calculated
            
            # Anomaly detection handled in analysis phase
            
            result[f"{i}"] = {
                "ads_eng": ads_energy,
                # Adslab displacement metrics
                "adslab_max_disp": ads_displacement_stats["max_disp"],
                "adslab_pos_mae": ads_displacement_stats["mae_mobile"],
                "adslab_pos_rmsd": ads_displacement_stats["rmsd_mobile"],
                # Energy changes
                "adslab_energy_change": ads_energy_change,
                # Timing and steps
                "adslab_time": ads_time,
                "adslab_steps": ads_step,
            }
            
            informs["ads_eng"].append(ads_energy)
            informs["adslab_max_disp"].append(ads_displacement_stats["max_disp"])
            informs["adslab_pos_mae"].append(ads_displacement_stats["mae_mobile"])
            informs["adslab_pos_rmsd"].append(ads_displacement_stats["rmsd_mobile"])
        
        ads_med_index, ads_med_eng = find_median_index(informs["ads_eng"])
        ads_eng_seed_range = np.max(np.array(informs["ads_eng"])) - np.min(np.array(informs["ads_eng"]))
        
        # Anomaly detection performed in analysis phase
        
        # Calculate efficiency metrics for OC20 mode
        # Get adslab atoms count (since we only calculate adslab in OC20 mode)
        adslab_key = [key for key in reaction_data["raw"].keys() if "star" in key and key != "star"][0]
        adslab_atoms = len(reaction_data["raw"][adslab_key]["atoms"])
        
        result["final"] = {
            "ads_eng_median": ads_med_eng,
            "median_num": ads_med_index,
            "adslab_max_disp": np.max(np.array(informs["adslab_max_disp"])),
            "ads_eng_seed_range": ads_eng_seed_range,
            "time_total_adslab": time_total_ads,
            "steps_total_adslab": steps_total_ads,
            "step_weighted_atoms": adslab_atoms,  # In OC20 mode, only adslab is calculated
            "time_per_step": time_total_ads / steps_total_ads if steps_total_ads > 0 else 0,
            "time_per_step_per_atom": time_total_ads / (steps_total_ads * adslab_atoms) if steps_total_ads > 0 else 0,  # Use actual adslab atom count
        }
        
        return {"reaction_result": result, "time_consumed": time_consumed}
    
    def _save_results_basic(self, save_directory, final_result, gas_energies, gas_energies_single, slab_energies, failures=None):
        """Save results for basic mode."""
        calculation_settings = get_calculation_settings(self.config)
        save_calculation_results(
            save_directory, self.mlip_name,
            final_result, gas_energies, gas_energies_single,
            calculation_settings, failures=failures
        )
        # Persist the slab-relaxation cache as a restart-safe sibling file (1.1.1),
        # mirroring the gas-cache persistence discipline.
        # Stamp the relaxation settings so a later run with different settings
        # invalidates the cache instead of reusing stale relaxations.
        slab_energies["__relax_config__"] = self._relax_sig()
        # Atomic write: a kill mid-write (e.g. SLURM walltime) must not truncate
        # the cache and lose restart-safety. Write to a temp file then rename.
        slab_cache_path = os.path.join(save_directory, f"{self.mlip_name}_slab_energies.json")
        tmp_path = slab_cache_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(slab_energies, f, cls=NumpyEncoder)
        os.replace(tmp_path, slab_cache_path)

    def _save_results_oc20(self, save_directory, final_result, failures=None):
        """Save results for OC20 mode."""
        calculation_settings = get_calculation_settings(self.config)
        save_calculation_results(
            save_directory, self.mlip_name,
            final_result, calculation_settings=calculation_settings,
            failures=failures
        )
    
    def _calculate_max_bond_change(self, initial_atoms, final_atoms, adsorbate_indices):
        """
        Calculate maximum bond length change percentage for adsorbate atoms.
        
        Args:
            initial_atoms: Initial ASE Atoms object
            final_atoms: Final ASE Atoms object after relaxation
            adsorbate_indices: List of indices for adsorbate atoms
            
        Returns:
            float: Maximum bond length change percentage, or 0 if no bonds found
        """
        if not adsorbate_indices:
            return 0.0
            
        max_change_pct = 0.0
        n_substrate_atoms = len(initial_atoms) - len(adsorbate_indices)
        
        # Get chemical bond cutoff and ensure it's saved in config
        cutoff = self.config.get("chemical_bond_cutoff", get_default("chemical_bond_cutoff", CALCULATION_DEFAULTS))
        self.config["chemical_bond_cutoff"] = cutoff
        
        for ads_idx in adsorbate_indices:
            # Find neighbors within cutoff distance
            
            for partner_idx in range(len(initial_atoms)):
                if partner_idx == ads_idx:
                    continue
                    
                # Calculate distances with PBC
                dist_initial = initial_atoms.get_distance(ads_idx, partner_idx, mic=True)
                dist_final = final_atoms.get_distance(ads_idx, partner_idx, mic=True)
                
                # Only consider if within cutoff in either initial or final
                if dist_initial <= cutoff or dist_final <= cutoff:
                    if dist_initial > 0:  # Avoid division by zero
                        change_pct = abs(dist_final - dist_initial) / dist_initial * 100
                        max_change_pct = max(max_change_pct, change_pct)
        
        return max_change_pct
    
    def _calculate_substrate_displacement(self, slab_initial, slab_final, adslab_initial, adslab_final, adsorbate_indices):
        """
        Calculate maximum substrate atom displacement in adslab.
        
        Args:
            slab_initial: Initial slab ASE Atoms object
            slab_final: Final slab ASE Atoms object after relaxation
            adslab_initial: Initial adslab ASE Atoms object
            adslab_final: Final adslab ASE Atoms object after relaxation
            adsorbate_indices: List of indices for adsorbate atoms
            
        Returns:
            float: Maximum substrate displacement in Angstroms
        """
        # NOTE: this metric depends ONLY on the adslab (initial vs final) and
        # adsorbate_indices -- slab_initial/slab_final are intentionally unused.
        # The adslab cache relies on this: a cached substrate_displacement is
        # reused for any reaction with an identical adslab regardless of its clean
        # slab. Do not start using the slab args here without revisiting that cache.
        # Substrate atoms are all non-adsorbate atoms, indexed in the adslab
        # space (not the slab space) so they are correct even when adsorbate
        # indices are not strictly trailing.
        substrate_indices = [i for i in range(len(adslab_initial)) if i not in adsorbate_indices]
        if not substrate_indices:
            return 0.0

        # Use ASE's minimum-image convention so displacements are correct for
        # non-orthogonal cells (e.g. hexagonal fcc(111)/hcp) and wrapped atoms.
        diffs = (adslab_final.get_positions()[substrate_indices]
                 - adslab_initial.get_positions()[substrate_indices])
        mic_diffs, _ = find_mic(diffs, adslab_initial.cell, adslab_initial.pbc)
        return float(np.linalg.norm(mic_diffs, axis=1).max())
 