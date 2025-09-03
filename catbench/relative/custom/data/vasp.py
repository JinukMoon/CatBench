"""
VASP Custom Relative Energy Data Preprocessing for CatBench.

This module provides functions for processing VASP calculation data for
custom relative energy benchmarking tasks with user-defined stoichiometry.
"""

import os
import json
import logging
import numpy as np
from collections import Counter
from ase.io import read
from catbench.utils.data_utils import save_catbench_json


def custom_vasp_preprocessing(dataset_name, coeff_setting, save_directory="raw_data"):
    """
    Process VASP data for custom relative energy benchmarking.
    
    Expected Directory Structure:
        dataset_name/
        ├── target/
        │   ├── target1/
        │   │   ├── CONTCAR
        │   │   └── OSZICAR
        │   └── target2/
        │       ├── CONTCAR
        │       └── OSZICAR
        └── reference/
            ├── ref1/
            │   ├── CONTCAR
            │   └── OSZICAR
            └── ref2/
                ├── CONTCAR
                └── OSZICAR
    
    Args:
        dataset_name (str): Path to the root directory containing VASP calculations
        coeff_setting (dict): Custom reaction configuration
                            Format: {
                                "reaction_name": {
                                    "target": coefficient,
                                    "ref1": coefficient,
                                    "ref2": coefficient,
                                    "normalization_factor": factor (default: 1)
                                }
                            }
        save_directory (str, optional): Directory to save output files. Default: "raw_data"
        
    Output Files:
        - {dataset_name}_custom.json: Processed custom relative energy data
        - {dataset_name}_custom_preprocessing.log: Processing details
        
    Returns:
        str: Path to the created JSON file
        
    Raises:
        FileNotFoundError: If required VASP files are missing
        ValueError: If coefficient setting is invalid
    """
    
    # Setup logging
    os.makedirs(save_directory, exist_ok=True)
    dataset_basename = os.path.basename(dataset_name.rstrip('/'))
    log_file = os.path.join(save_directory, f"{dataset_basename}_custom_preprocessing.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting VASP custom relative energy preprocessing for dataset: {dataset_name}")
    logger.info(f"Custom reactions: {coeff_setting}")
    
    if not os.path.exists(dataset_name):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_name}")
    
    target_dir = os.path.join(dataset_name, "target")
    reference_dir = os.path.join(dataset_name, "reference")
    
    if not (os.path.exists(target_dir) and os.path.exists(reference_dir)):
        raise FileNotFoundError("Both 'target' and 'reference' directories are required")
    
    custom_energy_data = {}
    
    for reaction_name, reaction_coeffs in coeff_setting.items():
        try:
            logger.info(f"Processing custom reaction: {reaction_name}")
            logger.info(f"  Coefficients: {reaction_coeffs}")
            
            # Extract target and reference names from coefficients
            target_coeff = reaction_coeffs.get("target")
            if target_coeff is None:
                raise ValueError(f"Missing 'target' coefficient for reaction {reaction_name}")
                
            # Find target name (assume single target with coefficient 1 or as specified)
            target_name = reaction_name  # Use reaction name as target name by default
            normalization_factor = reaction_coeffs.get("normalization_factor", 1)
            
            # Read target structure and energy
            target_path = os.path.join(target_dir, target_name)
            if not os.path.exists(target_path):
                raise FileNotFoundError(f"Target directory not found: {target_path}")
                
            target_contcar = os.path.join(target_path, "CONTCAR")
            target_oszicar = os.path.join(target_path, "OSZICAR")
            
            if not (os.path.exists(target_contcar) and os.path.exists(target_oszicar)):
                raise FileNotFoundError(f"Missing CONTCAR or OSZICAR in {target_path}")
                
            target_atoms = read(target_contcar)
            target_energy = _read_energy_from_oszicar(target_oszicar)
            target_composition = _get_composition(target_atoms)
            
            logger.info(f"  Target composition: {target_composition}")
            logger.info(f"  Normalization factor: {normalization_factor}")
            
            # Read reference structures and energies
            references = {}
            stoichiometry = {"target": target_coeff, "normalization_factor": normalization_factor}
            
            for coeff_name, coeff_value in reaction_coeffs.items():
                if coeff_name in ["target", "normalization_factor"]:
                    continue
                    
                ref_path = os.path.join(reference_dir, coeff_name)
                if not os.path.exists(ref_path):
                    raise FileNotFoundError(f"Reference directory not found: {ref_path}")
                    
                ref_contcar = os.path.join(ref_path, "CONTCAR")
                ref_oszicar = os.path.join(ref_path, "OSZICAR")
                
                if not (os.path.exists(ref_contcar) and os.path.exists(ref_oszicar)):
                    raise FileNotFoundError(f"Missing CONTCAR or OSZICAR in {ref_path}")
                    
                ref_atoms = read(ref_contcar)
                ref_energy = _read_energy_from_oszicar(ref_oszicar)
                ref_composition = _get_composition(ref_atoms)
                
                references[coeff_name] = {
                    "atoms": ref_atoms,
                    "energy": ref_energy,
                    "composition": ref_composition
                }
                
                stoichiometry[coeff_name] = coeff_value
                
                logger.info(f"  Reference {coeff_name}: {ref_composition}, coeff: {coeff_value}")
            
            # Store custom energy data
            custom_energy_data[reaction_name] = {
                "target": {
                    "atoms": target_atoms,
                    "energy": target_energy,
                    "composition": target_composition
                },
                "references": references,
                "stoichiometry": stoichiometry,
                "normalization_factor": normalization_factor,
                "source": "vasp"
            }
            
            logger.info(f"Successfully processed custom reaction: {reaction_name}")
            
        except Exception as e:
            logger.error(f"Error processing custom reaction {reaction_name}: {str(e)}")
            raise
    
    # Save processed data
    output_path = os.path.join(save_directory, f"{dataset_basename}_custom.json")
    save_catbench_json(custom_energy_data, output_path)
        
    logger.info(f"Successfully processed {len(custom_energy_data)} custom reactions")
    logger.info(f"Custom relative energy data saved to: {output_path}")
    
    return output_path


# Utility functions

def _read_energy_from_oszicar(oszicar_path):
    """Read final energy from OSZICAR file."""
    with open(oszicar_path, 'r') as f:
        lines = f.readlines()
    
    # Find the last line with E0
    for line in reversed(lines):
        if 'E0=' in line:
            energy_str = line.split('E0=')[1].split()[0]
            return float(energy_str)
    
    raise ValueError(f"Could not find E0 energy in {oszicar_path}")


def _get_composition(atoms):
    """Extract composition dictionary from ASE atoms object."""
    symbols = atoms.get_chemical_symbols()
    return dict(Counter(symbols))