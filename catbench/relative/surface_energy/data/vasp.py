"""
VASP Surface Energy Data Preprocessing for CatBench.

This module provides functions for processing VASP calculation data specifically for
surface energy benchmarking tasks.
"""

import os
import json
import logging
import numpy as np
from collections import Counter
from ase.io import read
from catbench.utils.data_utils import save_catbench_json, cleanup_vasp_files


def surface_energy_vasp_preprocessing(dataset_name, save_directory="raw_data"):
    """
    Process VASP data for surface energy benchmarking.
    
    Expected Directory Structure:
        dataset_name/
        ├── surface_type1/
        │   ├── slab/
        │   │   ├── CONTCAR
        │   │   └── OSZICAR
        │   └── bulk/
        │       ├── CONTCAR
        │       └── OSZICAR
        └── surface_type2/
            ├── slab/
            └── bulk/
    
    Args:
        dataset_name (str): Path to the root directory containing VASP calculations
        save_directory (str, optional): Directory to save output files. Default: "raw_data"
        
    Output Files:
        - {dataset_name}_surface_energy.json: Processed surface energy data
        - {dataset_name}_surface_energy_preprocessing.log: Processing details
        
    Returns:
        str: Path to the created JSON file
        
    Raises:
        ValueError: If surface and bulk compositions are not integer multiples
        FileNotFoundError: If required VASP files are missing
    """
    
    # Setup logging
    os.makedirs(save_directory, exist_ok=True)
    dataset_basename = os.path.basename(dataset_name.rstrip('/'))
    log_file = os.path.join(save_directory, f"{dataset_basename}_surface_energy_preprocessing.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting VASP surface energy preprocessing for dataset: {dataset_name}")
    
    if not os.path.exists(dataset_name):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_name}")
    
    # Clean up unnecessary VASP files
    logger.info("Cleaning up unnecessary VASP files (keeping CONTCAR and OSZICAR)...")
    cleanup_vasp_files(dataset_name, keep_files=["CONTCAR", "OSZICAR"], verbose=True)
    
    surface_energy_data = {}
    
    # Find all surface system directories
    for surface_dir in os.listdir(dataset_name):
        surface_path = os.path.join(dataset_name, surface_dir)
        
        if not os.path.isdir(surface_path):
            continue
            
        surface_structure_path = os.path.join(surface_path, "slab")
        bulk_structure_path = os.path.join(surface_path, "bulk")
        
        if not (os.path.exists(surface_structure_path) and os.path.exists(bulk_structure_path)):
            logger.warning(f"Skipping {surface_dir}: missing slab or bulk directory")
            continue
            
        try:
            logger.info(f"Processing surface system: {surface_dir}")
            
            # Read surface structure and energy
            surface_contcar = os.path.join(surface_structure_path, "CONTCAR")
            surface_oszicar = os.path.join(surface_structure_path, "OSZICAR")
            
            if not (os.path.exists(surface_contcar) and os.path.exists(surface_oszicar)):
                logger.warning(f"Skipping {surface_dir}/slab: missing CONTCAR or OSZICAR")
                continue
                
            surface_atoms = read(surface_contcar)
            surface_energy = _read_energy_from_oszicar(surface_oszicar)
            surface_composition = _get_composition(surface_atoms)
            
            # Read bulk structure and energy
            bulk_contcar = os.path.join(bulk_structure_path, "CONTCAR")
            bulk_oszicar = os.path.join(bulk_structure_path, "OSZICAR")
            
            if not (os.path.exists(bulk_contcar) and os.path.exists(bulk_oszicar)):
                logger.warning(f"Skipping {surface_dir}/bulk: missing CONTCAR or OSZICAR")
                continue
                
            bulk_atoms = read(bulk_contcar)
            bulk_energy = _read_energy_from_oszicar(bulk_oszicar)
            bulk_composition = _get_composition(bulk_atoms)
            
            logger.info(f"  Surface composition: {surface_composition}")
            logger.info(f"  Bulk composition: {bulk_composition}")
            
            # Find integer multiple relationship
            n_factor = _find_composition_multiple(surface_composition, bulk_composition)
            
            if n_factor is None:
                error_msg = (f"Surface and bulk compositions are not integer multiples for {surface_dir}. "
                           f"Surface: {surface_composition}, Bulk: {bulk_composition}")
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            logger.info(f"  Found n-factor = {n_factor}")
            
            # Store surface energy data
            surface_energy_data[surface_dir] = {
                "surface": {
                    "atoms": surface_atoms,
                    "energy": surface_energy,
                    "composition": surface_composition
                },
                "bulk": {
                    "atoms": bulk_atoms,
                    "energy": bulk_energy,
                    "composition": bulk_composition
                },
                "n_factor": n_factor,
                "source": "vasp"
            }
            
            logger.info(f"Successfully processed surface system: {surface_dir}")
            
        except Exception as e:
            logger.error(f"Error processing {surface_dir}: {str(e)}")
            raise
    
    if not surface_energy_data:
        raise ValueError(f"No valid surface energy systems found in dataset {dataset_name}")
        
    # Save processed data
    output_path = os.path.join(save_directory, f"{dataset_basename}_surface_energy.json")
    save_catbench_json(surface_energy_data, output_path)
        
    logger.info(f"Successfully processed {len(surface_energy_data)} surface energy systems")
    logger.info(f"Surface energy data saved to: {output_path}")
    
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


def _find_composition_multiple(surface_comp, bulk_comp):
    """
    Find integer multiple n such that surface_comp = n * bulk_comp.
    
    Returns:
        int or None: The multiple factor if found, None otherwise
    """
    if not bulk_comp:
        return None
        
    # Check if all elements in bulk are present in surface
    for element in bulk_comp:
        if element not in surface_comp:
            return None
            
    # Calculate potential n factors for each element
    n_factors = []
    for element in bulk_comp:
        if bulk_comp[element] == 0:
            continue
        n = surface_comp[element] / bulk_comp[element]
        if n != int(n) or n <= 0:
            return None
        n_factors.append(int(n))
        
    # Check if all n factors are the same
    if len(set(n_factors)) != 1:
        return None
        
    n = n_factors[0]
    
    # Verify that surface_comp = n * bulk_comp for all elements
    for element in surface_comp:
        expected_count = n * bulk_comp.get(element, 0)
        if surface_comp[element] != expected_count:
            return None
            
    return n