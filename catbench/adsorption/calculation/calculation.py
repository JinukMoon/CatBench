"""
Adsorption energy calculation module for CatBench.

This module provides the AdsorptionCalculation class for performing
adsorption energy benchmarking calculations with MLIPs.
"""

import json
import os
import shutil

import numpy as np
from ase.io import write

from catbench.config import CALCULATION_DEFAULTS, get_default
from catbench.utils.calculation_utils import (
    energy_cal_gas, energy_cal_single, energy_cal, 
    calc_displacement, find_median_index, fix_z, NumpyEncoder
)
from catbench.utils.io_utils import (
    create_calculation_directories, get_result_directory, get_raw_data_path,
    load_existing_results, save_calculation_results, get_calculation_settings
)


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
        rate (float, optional): Fraction of atoms to fix (z-direction). Default: 0.5
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
                - rate: Learning rate for optimization
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
        # Apply configuration defaults
        for key, default_value in CALCULATION_DEFAULTS.items():
            if key not in self.config:
                self.config[key] = get_default(key, CALCULATION_DEFAULTS)
    
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
        create_calculation_directories(save_directory)
        return save_directory
    
    def _load_existing_results(self, save_directory):
        """Load existing results if available (automatic restart)."""
        try:
            final_result, gas_energies, gas_energies_single = load_existing_results(
                save_directory, self.mlip_name
            )
            return final_result, gas_energies, gas_energies_single
                
        except FileNotFoundError:
            print("Beginning calculation from scratch.")
            return {}, {}, {}
    
    def _run_basic(self):
        """Run basic benchmarking mode (full features with multiple calculators)."""
        ref_data = self._load_data()
        save_directory = self._setup_directories()
        final_result, gas_energies, gas_energies_single = self._load_existing_results(save_directory)
        
        print("Starting calculations...")
        for index, key in enumerate(ref_data):
            # Skip if already calculated
            if key in final_result:
                print(f"Skipping already calculated {key}")
                continue
                
            # Clean up any incomplete attempts
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
                result = self._process_reaction_basic(key, ref_data[key], save_directory, gas_energies, gas_energies_single)
                final_result[key] = result["reaction_result"]
                
                # Save results every save_step calculations
                if len(final_result) % self.config["save_step"] == 0:
                    print(f"💾 Saving results at {len(final_result)} calculations...")
                    self._save_results_basic(save_directory, final_result, gas_energies, gas_energies_single)
                
            except Exception as e:
                print(f"Error occurred while processing {key}: {str(e)}")
                print("Skipping to next reaction...")
                continue
        
        # Final save to ensure all results are saved
        print(f"💾 Final save: {len(final_result)} total calculations")
        self._save_results_basic(save_directory, final_result, gas_energies, gas_energies_single)
        
        print(f"{self.mlip_name} Benchmarking Finish")
        return save_directory
    
    def _process_reaction_basic(self, key, reaction_data, save_directory, gas_energies, gas_energies_single):
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
                energy_calculated = energy_cal_single(self.calculators[0], POSCAR_str)
                ads_energy_single += energy_calculated * reaction_data["raw"][structure]["stoi"]
                
                if structure == "star":
                    slab_energy_single = energy_calculated
                else:
                    adslab_energy_single = energy_calculated
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
        
        # Setup paths
        traj_path = f"{save_directory}/traj/{key}"
        log_path = f"{save_directory}/log/{key}"
        os.makedirs(traj_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        
        # Calculate z_target for fixing atoms
        POSCAR_star = reaction_data["raw"]["star"]["atoms"]
        z_target = fix_z(POSCAR_star, self.config["rate"])
        
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
        
        # Run calculations for each calculator
        for i in range(len(self.calculators)):
            ads_energy_calc = 0
            
            # Store initial and final structures for analysis
            slab_initial = None
            slab_final = None
            adslab_initial = None
            adslab_final = None
            
            for structure in reaction_data["raw"]:
                if "gas" not in str(structure):
                    POSCAR_str = reaction_data["raw"][structure]["atoms"]
                    (
                        energy_calculated,
                        steps_calculated,
                        CONTCAR_calculated,
                        time_calculated,
                        energy_change,
                    ) = energy_cal(
                        self.calculators[i],
                        POSCAR_str,
                        self.config["f_crit_relax"],
                        self.config["n_crit_relax"],
                        self.config["damping"],
                        z_target,
                        self.config["optimizer"],
                        f"{log_path}/{structure}_{i}.txt",
                        f"{traj_path}/{structure}_{i}",
                    )
                    
                    ads_energy_calc += energy_calculated * reaction_data["raw"][structure]["stoi"]
                    time_consumed += time_calculated
                    
                    if structure == "star":
                        slab_steps = steps_calculated
                        slab_displacement_stats = calc_displacement(POSCAR_str, CONTCAR_calculated, z_target)
                        slab_energy = energy_calculated
                        slab_time = time_calculated
                        slab_energy_change = energy_change
                        time_total_slab += time_calculated
                        steps_total_slab += steps_calculated
                        # Store slab structures
                        slab_initial = POSCAR_str.copy()
                        slab_final = CONTCAR_calculated.copy()
                    else:
                        ads_step = steps_calculated
                        ads_displacement_stats = calc_displacement(POSCAR_str, CONTCAR_calculated, z_target)
                        ads_energy = energy_calculated
                        ads_time = time_calculated
                        ads_energy_change = energy_change
                        time_total_ads += time_calculated
                        steps_total_ads += steps_calculated
                        # Store adslab structures
                        adslab_initial = POSCAR_str.copy()
                        adslab_final = CONTCAR_calculated.copy()
                        
                else:  # Gas molecule
                    gas_tag = f"{structure}_{i}th"
                    if gas_tag in gas_energies:
                        ads_energy_calc += gas_energies[gas_tag] * reaction_data["raw"][structure]["stoi"]
                    else:
                        print(f"{gas_tag} calculating")
                        gas_CONTCAR, gas_energy = energy_cal_gas(
                            self.calculators[i],
                            reaction_data["raw"][structure]["atoms"],
                            self.config["f_crit_relax"],
                            f"{save_directory}/gases/POSCARs/POSCAR_{gas_tag}",
                            self.config["optimizer"],
                            f"{save_directory}/gases/log/{gas_tag}.txt",
                            f"{save_directory}/gases/traj/{gas_tag}",
                        )
                        gas_energies[gas_tag] = gas_energy
                        ads_energy_calc += gas_energy * reaction_data["raw"][structure]["stoi"]
                        write(f"{save_directory}/gases/CONTCARs/CONTCAR_{gas_tag}", gas_CONTCAR)
            
            # Calculate bond change and substrate displacement
            max_bond_change = 0.0
            substrate_disp = 0.0
            
            if adslab_initial is not None and adslab_final is not None and adsorbate_indices:
                # Calculate max bond change for adsorbate
                max_bond_change = self._calculate_max_bond_change(
                    adslab_initial, adslab_final, adsorbate_indices
                )
                
                # Calculate substrate displacement in adslab
                if slab_initial is not None and slab_final is not None:
                    substrate_disp = self._calculate_substrate_displacement(
                        slab_initial, slab_final, adslab_initial, adslab_final, adsorbate_indices
                    )
            
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
        final_result, _, _ = self._load_existing_results(save_directory)  # gas_energies not needed for OC20
        
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
                    print(f"💾 Saving results at {len(final_result)} calculations...")
                    self._save_results_oc20(save_directory, final_result)
                
            except Exception as e:
                print(f"Error occurred while processing {key}: {str(e)}")
                print("Skipping to next reaction...")
                continue
        
        # Final save to ensure all results are saved
        print(f"💾 Final save: {len(final_result)} total calculations")
        self._save_results_oc20(save_directory, final_result)
        
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
        
        # Setup paths
        traj_path = f"{save_directory}/traj/{key}"
        log_path = f"{save_directory}/log/{key}"
        os.makedirs(traj_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        
        POSCAR_star = reaction_data["raw"]["star"]["atoms"]
        z_target = fix_z(POSCAR_star, self.config["rate"])
        
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
                        z_target,
                        self.config["optimizer"],
                        f"{log_path}/{structure}_{i}.txt",
                        f"{traj_path}/{structure}_{i}",
                    )
                    time_consumed += time_calculated
                    
                    ads_step = steps_calculated
                    ads_displacement_stats = calc_displacement(POSCAR_str, CONTCAR_calculated, z_target)
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
    
    def _save_results_basic(self, save_directory, final_result, gas_energies, gas_energies_single):
        """Save results for basic mode."""
        calculation_settings = get_calculation_settings(self.config)
        save_calculation_results(
            save_directory, self.mlip_name,
            final_result, gas_energies, gas_energies_single,
            calculation_settings
        )
    
    def _save_results_oc20(self, save_directory, final_result):
        """Save results for OC20 mode."""
        calculation_settings = get_calculation_settings(self.config)
        save_calculation_results(
            save_directory, self.mlip_name,
            final_result, calculation_settings=calculation_settings
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
        n_substrate = len(slab_initial)
        
        # Get substrate indices (all non-adsorbate atoms)
        substrate_indices = [i for i in range(n_substrate) if i not in adsorbate_indices]
        
        if not substrate_indices:
            return 0.0
            
        max_disp = 0.0
        
        for idx in substrate_indices:
            # Calculate displacement with PBC
            pos_initial = adslab_initial[idx].position
            pos_final = adslab_final[idx].position
            
            # Use cell to handle PBC properly
            cell = adslab_initial.cell
            diff = pos_final - pos_initial
            
            # Apply minimum image convention
            for i in range(3):
                if adslab_initial.pbc[i]:
                    diff[i] = diff[i] - cell[i, i] * np.round(diff[i] / cell[i, i])
            
            displacement = np.linalg.norm(diff)
            max_disp = max(max_disp, displacement)
        
        return max_disp
 