"""
Relative Energy Calculation Module for CatBench.

This module provides the RelativeEnergyCalculation class for performing
single point energy calculations for relative energy benchmarking tasks.
"""

import os
import json
import numpy as np
from ase.io import write
from catbench.utils.io_utils import get_result_directory, get_raw_data_path, save_json
from catbench.utils.data_utils import load_catbench_json


class RelativeEnergyCalculation:
    """
    Relative Energy Calculation class for MLIP benchmarking.
    
    This class performs single point energy calculations for relative energy
    benchmarking tasks including surface energy, bulk formation energy, and
    custom relative energy calculations.
    
    Task Types:
        surface: Surface energy calculations (surface_energy - n*bulk_energy) / (2*A)
        bulk_formation: Bulk formation energy calculations per atom
        custom: Custom relative energy calculations with user-defined stoichiometry
    
    Args:
        calculator: Single ASE-compatible MLIP calculator for benchmarking
        **config: Configuration parameters:
            - task_type (str): Type of relative energy calculation ("surface", "bulk_formation", "custom")
            - mlip_name (str): Name identifier for the MLIP being benchmarked
            - benchmark (str): Name of the benchmark dataset (pickle file basename)
        
    Raises:
        ValueError: If task_type is not valid or required parameters are missing
        FileNotFoundError: If benchmark data file is not found
    """
    
    def __init__(self, calculator, **config):
        self.calculator = calculator
        
        # Validate required config parameters
        required_keys = ["mlip_name", "benchmark", "task_type"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config parameter: {key}")
        
        # Validate task type
        task_type = config["task_type"]
        valid_tasks = ["surface", "bulk_formation", "custom"]
        if task_type not in valid_tasks:
            raise ValueError(f"task_type must be one of {valid_tasks}, got: {task_type}")
        
        self.task_type = task_type
        self.config = config
        self.mlip_name = config["mlip_name"]
        self.benchmark = config["benchmark"]
        
        # Load data
        self.data = self._load_data()
        
        print(f"Initialized RelativeEnergyCalculation for {task_type} task")
        print(f"MLIP: {self.mlip_name}, Benchmark: {self.benchmark}")
        print(f"Found {len(self.data)} systems to calculate")
    
    def _load_data(self):
        """Load relative energy data from JSON file."""
        json_suffixes = {
            "surface": "_surface_energy.json",
            "bulk_formation": "_bulk_formation.json",
            "custom": "_custom.json"
        }
        json_name = f"{self.benchmark}{json_suffixes[self.task_type]}"
            
        path_json = os.path.join(os.getcwd(), f"raw_data/{json_name}")
        
        if not os.path.exists(path_json):
            raise FileNotFoundError(f"Benchmark data file not found: {path_json}")
            
        return load_catbench_json(path_json)
    
    def run(self):
        """
        Run the relative energy calculations.
        
        Returns:
            str: Path to the results directory
        """
        task_descriptions = {
            "surface": "surface energy benchmark",
            "bulk_formation": "bulk formation energy benchmark", 
            "custom": "custom relative energy benchmark"
        }
        task_desc = task_descriptions.get(self.task_type, self.task_type)
        print(f"Starting {self.mlip_name} Relative Energy Calculation ({task_desc})")
        
        # Setup directories
        save_directory = self._setup_directories()
        
        # Process each system
        results = {}
        for i, (system_name, system_data) in enumerate(self.data.items()):
            print(f"[{i+1}/{len(self.data)}] Processing {system_name}")
            
            try:
                if self.task_type == "surface":
                    result = self._calculate_surface_energy(system_name, system_data, save_directory)
                elif self.task_type == "bulk_formation":
                    result = self._calculate_bulk_formation_energy(system_name, system_data, save_directory)
                else:  # custom
                    result = self._calculate_custom_energy(system_name, system_data, save_directory)
                    
                results[system_name] = result
                
            except Exception as e:
                print(f"Error calculating {system_name}: {str(e)}")
                continue
        
        # Save results
        self._save_results(save_directory, results)
        
        print(f"{self.mlip_name} Relative Energy Calculation Complete ({task_desc})")
        return save_directory
    
    def _setup_directories(self):
        """Setup output directories for relative energy results."""
        save_directory = os.path.join(os.getcwd(), "result", self.mlip_name)
        os.makedirs(save_directory, exist_ok=True)
        return save_directory
    
    def _calculate_surface_energy(self, system_name, system_data, save_directory):
        """Calculate surface energy for a surface-bulk pair."""
        # Handle both CatHub (star) and VASP (surface) naming
        surface_key = "star" if "star" in system_data else "surface"

        surface_atoms = system_data[surface_key]["atoms"]
        bulk_atoms = system_data["bulk"]["atoms"]
        n_factor = system_data["n_factor"]
        
        # Reference surface energy calculation
        ref_surface_total_energy = system_data[surface_key]["energy"]
        ref_bulk_total_energy = system_data["bulk"]["energy"]
        surface_area = self._calculate_surface_area(surface_atoms)
        ref_surface_energy = (ref_surface_total_energy - n_factor * ref_bulk_total_energy) / (2 * surface_area)
        
        surface_atoms.calc = self.calculator
        mlip_surface_total_energy = surface_atoms.get_potential_energy()
        
        bulk_atoms.calc = self.calculator
        mlip_bulk_total_energy = bulk_atoms.get_potential_energy()
        
        # Surface energy calculation
        mlip_surface_energy = (mlip_surface_total_energy - n_factor * mlip_bulk_total_energy) / (2 * surface_area)

        # Save structures directly in save_directory
        # Save structures
        surface_file = os.path.join(save_directory, f"{system_name}.xyz")
        bulk_system_name = system_name.replace("_surface", "_bulk") 
        bulk_file = os.path.join(save_directory, f"{bulk_system_name}.xyz")
        write(surface_file, surface_atoms, plain=True)
        write(bulk_file, bulk_atoms, plain=True)
        
        return {
            "reference": {
                "surface_total_energy": ref_surface_total_energy,
                "bulk_total_energy": ref_bulk_total_energy,
                "surface_energy": ref_surface_energy,
                "surface_area": surface_area,
                "n_factor": n_factor
            },
            "mlip_calculation": {
                "surface_total_energy": float(mlip_surface_total_energy),
                "bulk_total_energy": float(mlip_bulk_total_energy),
                "surface_energy": float(mlip_surface_energy),
            }
        }
    
    def _calculate_bulk_formation_energy(self, system_name, system_data, save_directory):
        """Calculate bulk formation energy for a target-references system."""
        result = self._calculate_formation_energy(system_name, system_data, save_directory)
        result["reference"]["formation_energy"] = result["reference"].pop("relative_energy")
        result["mlip_calculation"]["formation_energy"] = result["mlip_calculation"].pop("relative_energy")
        result["task_type"] = "bulk_formation"
        return result
    
    def _calculate_custom_energy(self, system_name, system_data, save_directory):
        """Calculate custom relative energy for a target-references system."""
        result = self._calculate_formation_energy(system_name, system_data, save_directory)
        result["reference"]["custom_energy"] = result["reference"].pop("relative_energy")
        result["mlip_calculation"]["custom_energy"] = result["mlip_calculation"].pop("relative_energy")
        result["task_type"] = "custom"
        return result
    
    def _calculate_formation_energy(self, system_name, system_data, save_directory):
        """Common calculation for formation-type energies (bulk_formation and custom)."""
        target_atoms = system_data["target"]["atoms"].copy()
        references = system_data["references"]
        stoichiometry = system_data["stoichiometry"]
        normalization_factor = system_data["normalization_factor"]
        
        # Reference energy calculation
        ref_target_energy = system_data["target"]["energy"]
        ref_relative_energy = stoichiometry["target"] * ref_target_energy
        
        for ref_name, ref_data in references.items():
            ref_relative_energy += stoichiometry[ref_name] * ref_data["energy"]
        
        ref_relative_energy /= normalization_factor
        
        # MLIP calculation
        target_atoms_calc = target_atoms.copy()
        target_atoms_calc.calc = self.calculator
        mlip_target_energy = target_atoms_calc.get_potential_energy()
        
        # Reference calculations
        mlip_ref_energies = {}
        for ref_name, ref_data in references.items():
            ref_atoms_calc = ref_data["atoms"].copy()
            ref_atoms_calc.calc = self.calculator
            mlip_ref_energies[ref_name] = ref_atoms_calc.get_potential_energy()
            
            # Save reference structure directly in save_directory
            ref_file = os.path.join(save_directory, f"{system_name}_{ref_name}.extxyz")
            write(ref_file, ref_atoms_calc)
        
        # Relative energy calculation
        mlip_relative_energy = stoichiometry["target"] * mlip_target_energy
        for ref_name in references.keys():
            mlip_relative_energy += stoichiometry[ref_name] * mlip_ref_energies[ref_name]
        mlip_relative_energy /= normalization_factor
        
        # Save target structure directly in save_directory
        target_file = os.path.join(save_directory, f"{system_name}_target.extxyz")
        write(target_file, target_atoms_calc)
        
        return {
            "reference": {
                "target_energy": ref_target_energy,
                "reference_energies": {name: ref["energy"] for name, ref in references.items()},
                "relative_energy": ref_relative_energy,
                "stoichiometry": stoichiometry,
                "normalization_factor": normalization_factor
            },
            "mlip_calculation": {
                "target_energy": mlip_target_energy,
                "reference_energies": mlip_ref_energies,
                "relative_energy": mlip_relative_energy
            }
        }
    
    def _calculate_surface_area(self, atoms):
        """Calculate surface area from unit cell vectors a and b."""
        cell = atoms.get_cell()
        a_vector = cell[0]
        b_vector = cell[1]
        
        # Surface area = |a × b|
        cross_product = np.cross(a_vector, b_vector)
        surface_area = np.linalg.norm(cross_product)
        
        return surface_area
    
    def _save_results(self, save_directory, results):
        """Save calculation results to JSON file."""
        output_suffixes = {
            "surface": "_surface_energy_result.json",
            "bulk_formation": "_bulk_formation_result.json",
            "custom": "_custom_result.json"
        }
        output_file = f"{self.mlip_name}{output_suffixes[self.task_type]}"
        
        output_path = os.path.join(save_directory, output_file)
        
        # Add calculation metadata
        results_with_metadata = {
            "task_type": self.task_type,
            "mlip_name": self.mlip_name,
            "benchmark": self.benchmark,
            "results": results
        }
        
        save_json(results_with_metadata, output_path)
        
        print(f"Results saved to: {output_path}")
    
 