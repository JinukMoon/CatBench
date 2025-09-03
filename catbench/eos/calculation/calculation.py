"""
EOS Calculation Module for CatBench.

This module performs single point energy calculations for EOS benchmarking.
"""

import os
import json
import io
import numpy as np
from ase.io import read, write
from catbench.utils.calculation_utils import energy_cal_single
from catbench.utils.io_utils import save_json


class EOSCalculation:
    """
    EOS Calculation class for MLIP benchmarking.
    
    Performs single point energy calculations at different volumes for
    Equation of State (EOS) analysis.
    
    Args:
        calculator: ASE-compatible MLIP calculator
        mlip_name (str): Name identifier for the MLIP
        benchmark (str): Name of the benchmark dataset
    """
    
    def __init__(self, calculator, mlip_name, benchmark):
        """Initialize EOS calculation."""
        self.calculator = calculator
        self.mlip_name = mlip_name
        self.benchmark = benchmark
        
        # Load EOS data
        self.data = self._load_data()
        
        print(f"Initialized EOSCalculation")
        print(f"MLIP: {self.mlip_name}, Benchmark: {self.benchmark}")
        print(f"Found {len(self.data)} materials to calculate")
    
    def _load_data(self):
        """Load EOS data from JSON file."""
        json_name = f"{self.benchmark}_eos.json"
        path_json = os.path.join(os.getcwd(), f"raw_data/{json_name}")
        
        if not os.path.exists(path_json):
            raise FileNotFoundError(f"Benchmark data file not found: {path_json}")
        
        # Load JSON and convert atoms_json back to Atoms objects
        with open(path_json, 'r') as f:
            data = json.load(f)
        
        # Convert atoms_json strings back to Atoms objects
        for material, material_data in data.items():
            for point in material_data["points"]:
                if "atoms_json" in point:
                    # Convert JSON string to Atoms object
                    buffer = io.StringIO(point["atoms_json"])
                    atoms = read(buffer, format='json')
                    point["atoms"] = atoms
                    del point["atoms_json"]
        
        return data
    
    def run(self):
        """
        Run EOS calculations for all materials and volumes.
        
        Returns:
            str: Path to the results directory
        """
        print(f"Starting {self.mlip_name} EOS Calculation")
        
        # Setup directories
        save_directory = self._setup_directories()
        
        # Results dictionary
        results = {}
        
        # Process each material
        for material_name, material_data in self.data.items():
            print(f"\nProcessing material: {material_name}")
            
            material_results = {
                "points": [],
                "vasp_min_energy": material_data["min_energy"],
                "vasp_equilibrium_volume": material_data["equilibrium_volume"]
            }
            
            # Create material subdirectory
            material_dir = os.path.join(save_directory, material_name)
            os.makedirs(material_dir, exist_ok=True)
            
            # Process each volume point
            points = material_data["points"]
            for i, point in enumerate(points):
                volume = point["volume"]
                atoms = point["atoms"].copy()
                vasp_energy = point["energy"]
                folder_name = point["folder"]
                
                print(f"  Point {i+1}/{len(points)}: V={volume:.2f} Å³", end="")
                
                # Single point energy calculation
                try:
                    mlip_energy = energy_cal_single(self.calculator, atoms)
                    
                    # Save structure with energy in material subdirectory
                    atoms_with_calc = atoms.copy()
                    atoms_with_calc.calc = self.calculator
                    structure_file = os.path.join(material_dir, f"{folder_name}.extxyz")
                    write(structure_file, atoms_with_calc)
                    
                    # Store results
                    material_results["points"].append({
                        "folder": folder_name,
                        "volume": volume,
                        "mlip_energy": mlip_energy,
                        "vasp_energy": vasp_energy,
                        "n_atoms": len(atoms)
                    })
                    
                    print(f" -> E_mlip={mlip_energy:.4f} eV, E_vasp={vasp_energy:.4f} eV")
                    
                except Exception as e:
                    print(f" -> Error: {str(e)}")
                    material_results["points"].append({
                        "folder": folder_name,
                        "volume": volume,
                        "mlip_energy": None,
                        "vasp_energy": vasp_energy,
                        "n_atoms": len(atoms),
                        "error": str(e)
                    })
            
            # Calculate MLIP equilibrium if all calculations succeeded
            valid_points = [p for p in material_results["points"] if p["mlip_energy"] is not None]
            if valid_points:
                mlip_energies = [p["mlip_energy"] for p in valid_points]
                min_idx = np.argmin(mlip_energies)
                material_results["mlip_min_energy"] = mlip_energies[min_idx]
                material_results["mlip_equilibrium_volume"] = valid_points[min_idx]["volume"]
                material_results["n_valid_points"] = len(valid_points)
                
                print(f"  MLIP equilibrium: V={material_results['mlip_equilibrium_volume']:.2f} Å³, "
                      f"E={material_results['mlip_min_energy']:.4f} eV")
                print(f"  VASP equilibrium: V={material_results['vasp_equilibrium_volume']:.2f} Å³, "
                      f"E={material_results['vasp_min_energy']:.4f} eV")
            else:
                material_results["mlip_min_energy"] = None
                material_results["mlip_equilibrium_volume"] = None
                material_results["n_valid_points"] = 0
                print(f"  Warning: No valid MLIP calculations for {material_name}")
            
            results[material_name] = material_results
        
        # Save results
        self._save_results(save_directory, results)
        
        print(f"\n{self.mlip_name} EOS Calculation Complete")
        print(f"Results saved to: {save_directory}")
        
        return save_directory
    
    def _setup_directories(self):
        """Setup output directories for EOS results."""
        save_directory = os.path.join(os.getcwd(), f"result", self.mlip_name)
        os.makedirs(save_directory, exist_ok=True)
        return save_directory
    
    def _save_results(self, save_directory, results):
        """Save calculation results to JSON file."""
        output_file = f"{self.mlip_name}_eos_result.json"
        output_path = os.path.join(save_directory, output_file)
        
        # Add metadata
        results_with_metadata = {
            "mlip_name": self.mlip_name,
            "benchmark": self.benchmark,
            "results": results
        }
        
        save_json(results_with_metadata, output_path)
        
        print(f"Results saved to: {output_path}")