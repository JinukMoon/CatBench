"""
VASP Bulk Formation Energy Data Preprocessing for CatBench.

This module provides functions for processing VASP calculation data specifically for
bulk formation energy benchmarking tasks.
"""

import os
import json
import logging
import numpy as np
from collections import Counter
from ase.io import read
from catbench.utils.data_utils import save_catbench_json, cleanup_vasp_files


def bulk_formation_vasp_preprocessing(dataset_name, coeff_setting, save_directory="raw_data"):
    """
    Process VASP data for bulk formation energy benchmarking.
    
    Expected Directory Structure:
        dataset_name/
        ├── target/
        │   ├── target_bulk1/
        │   │   ├── CONTCAR
        │   │   └── OSZICAR
        │   └── target_bulk2/
        │       ├── CONTCAR
        │       └── OSZICAR
        └── reference/
            ├── ref_bulk1/
            │   ├── CONTCAR
            │   └── OSZICAR
            └── ref_bulk2/
                ├── CONTCAR
                └── OSZICAR
    
    Args:
        dataset_name (str): Path to the root directory containing VASP calculations
        coeff_setting (dict): Formation reactions configuration
                            Format: {"target_name": ["ref1", "ref2", ...]}
                            Example: {"Co2MnO4": ["Co_bulk", "Mn_bulk", "O2_bulk"]}
        save_directory (str, optional): Directory to save output files. Default: "raw_data"
        
    Output Files:
        - {dataset_name}_bulk_formation.json: Processed bulk formation data
        - {dataset_name}_bulk_formation_preprocessing.log: Processing details
        
    Returns:
        str: Path to the created JSON file
        
    Raises:
        ValueError: If stoichiometry cannot be balanced or references are insufficient
        FileNotFoundError: If required VASP files are missing
    """
    
    # Setup logging
    os.makedirs(save_directory, exist_ok=True)
    dataset_basename = os.path.basename(dataset_name.rstrip('/'))
    log_file = os.path.join(save_directory, f"{dataset_basename}_bulk_formation_preprocessing.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting VASP bulk formation preprocessing for dataset: {dataset_name}")
    logger.info(f"Formation reactions: {coeff_setting}")
    
    if not os.path.exists(dataset_name):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_name}")
    
    # Clean up unnecessary VASP files
    logger.info("Cleaning up unnecessary VASP files (keeping CONTCAR and OSZICAR)...")
    cleanup_vasp_files(dataset_name, keep_files=["CONTCAR", "OSZICAR"], verbose=True)
    
    target_dir = os.path.join(dataset_name, "target")
    reference_dir = os.path.join(dataset_name, "reference")
    
    if not (os.path.exists(target_dir) and os.path.exists(reference_dir)):
        raise FileNotFoundError("Both 'target' and 'reference' directories are required")
    
    formation_energy_data = {}
    
    for target_name, reference_list in coeff_setting.items():
        try:
            logger.info(f"Processing formation reaction: {target_name}")
            logger.info(f"  Required references: {reference_list}")
            
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
            normalization_factor = len(target_atoms)
            
            logger.info(f"  Target composition: {target_composition}")
            logger.info(f"  Normalization factor: {normalization_factor} atoms")
            
            # Read reference structures and energies
            references = {}
            for ref_name in reference_list:
                ref_path = os.path.join(reference_dir, ref_name)
                if not os.path.exists(ref_path):
                    raise FileNotFoundError(f"Reference directory not found: {ref_path}")
                    
                ref_contcar = os.path.join(ref_path, "CONTCAR")
                ref_oszicar = os.path.join(ref_path, "OSZICAR")
                
                if not (os.path.exists(ref_contcar) and os.path.exists(ref_oszicar)):
                    raise FileNotFoundError(f"Missing CONTCAR or OSZICAR in {ref_path}")
                    
                ref_atoms = read(ref_contcar)
                ref_energy = _read_energy_from_oszicar(ref_oszicar)
                ref_composition = _get_composition(ref_atoms)
                
                references[ref_name] = {
                    "atoms": ref_atoms,
                    "energy": ref_energy,
                    "composition": ref_composition
                }
                
                logger.info(f"  Reference {ref_name}: {ref_composition}")
            
            # Solve stoichiometry equations
            stoich_coeffs = _solve_formation_stoichiometry(target_composition, 
                                                          {name: ref["composition"] for name, ref in references.items()})
            
            logger.info(f"  Calculated stoichiometry: {stoich_coeffs}")
            
            # Store formation energy data
            formation_energy_data[target_name] = {
                "target": {
                    "atoms": target_atoms,
                    "energy": target_energy,
                    "composition": target_composition
                },
                "references": references,
                "stoichiometry": stoich_coeffs,
                "normalization_factor": normalization_factor,
                "source": "vasp"
            }
            
            logger.info(f"Successfully processed formation reaction: {target_name}")
            
        except Exception as e:
            logger.error(f"Error processing formation reaction {target_name}: {str(e)}")
            raise
    
    # Save processed data
    output_path = os.path.join(save_directory, f"{dataset_basename}_bulk_formation.json")
    save_catbench_json(formation_energy_data, output_path)
        
    logger.info(f"Successfully processed {len(formation_energy_data)} formation reactions")
    logger.info(f"Bulk formation data saved to: {output_path}")
    
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


def _solve_formation_stoichiometry(target_composition, reference_compositions):
    """
    Solve linear equations to find stoichiometry coefficients for formation reaction.
    
    Args:
        target_composition (dict): Target bulk composition
        reference_compositions (dict): Reference bulk compositions
        
    Returns:
        dict: Stoichiometry coefficients
        
    Raises:
        ValueError: If no solution or infinite solutions exist
    """
    
    # Get all unique elements
    all_elements = set(target_composition.keys())
    for ref_comp in reference_compositions.values():
        all_elements.update(ref_comp.keys())
    all_elements = sorted(list(all_elements))
    
    # Build coefficient matrix A and target vector b
    # target_composition = sum(coeff[ref] * reference_compositions[ref])
    ref_names = list(reference_compositions.keys())
    A = []
    b = []
    
    for element in all_elements:
        row = []
        for ref_name in ref_names:
            ref_comp = reference_compositions[ref_name]
            row.append(ref_comp.get(element, 0))
        A.append(row)
        b.append(target_composition.get(element, 0))
    
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    try:
        # Solve using least squares (handles overdetermined systems)
        coeffs, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Check if solution is valid (all coefficients should be reasonable)
        if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
            raise ValueError("No solution: Invalid coefficients obtained")
        
        # Check residuals for exact solution
        if len(residuals) > 0 and np.any(residuals > 1e-10):
            raise ValueError("No solution: Cannot balance stoichiometry with given references")
        
        # Check if solution is exact by verifying the equation
        reconstructed = A @ coeffs
        if not np.allclose(reconstructed, b, atol=1e-10):
            raise ValueError("No solution: Cannot balance stoichiometry with given references")
            
        # Check for ambiguous solutions (rank deficiency)
        if rank < len(ref_names) and len(ref_names) > 1:
            raise ValueError("Infinite solutions: Ambiguous stoichiometry (e.g., O2 and O4 both present)")
        
        # Build result dictionary
        result = {"target": 1}
        for i, ref_name in enumerate(ref_names):
            result[ref_name] = -float(coeffs[i])  # Negative because references are on reactant side
        
        return result
        
    except np.linalg.LinAlgError:
        raise ValueError("No solution: Linear algebra error in stoichiometry calculation")