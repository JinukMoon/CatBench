"""
VASP EOS Data Preprocessing for CatBench.

This module processes VASP calculation data for Equation of State (EOS) benchmarking.
"""

import os
import json
import logging
import io
import numpy as np
from ase.io import read, write


def eos_vasp_preprocessing(dataset_name, save_directory="raw_data"):
    """
    Process VASP data for EOS benchmarking.
    
    Expected Directory Structure:
        dataset_name/
        ├── material1/
        │   ├── 0/
        │   │   ├── CONTCAR
        │   │   └── OSZICAR
        │   ├── 1/
        │   │   ├── CONTCAR
        │   │   └── OSZICAR
        │   └── ... (up to 10)
        └── material2/
            └── ...
    
    Args:
        dataset_name (str): Path to the root directory containing VASP calculations
        save_directory (str, optional): Directory to save output files. Default: "raw_data"
        
    Output Files:
        - {dataset_name}_eos.json: Processed EOS data sorted by volume
        - {dataset_name}_eos_preprocessing.log: Processing details
        
    Returns:
        str: Path to the created JSON file
    """
    
    # Setup logging
    os.makedirs(save_directory, exist_ok=True)
    dataset_basename = os.path.basename(dataset_name.rstrip('/'))
    log_file = os.path.join(save_directory, f"{dataset_basename}_eos_preprocessing.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting VASP EOS preprocessing for dataset: {dataset_name}")
    
    if not os.path.exists(dataset_name):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_name}")
    
    eos_data = {}
    
    # Find all material directories
    for material_dir in os.listdir(dataset_name):
        material_path = os.path.join(dataset_name, material_dir)
        
        if not os.path.isdir(material_path):
            continue
            
        logger.info(f"Processing material: {material_dir}")
        
        # Lists to store volume-energy pairs
        volume_energy_pairs = []
        
        # Read all volume folders (could be 0-10 or any other naming)
        volume_folders = []
        for folder in os.listdir(material_path):
            folder_path = os.path.join(material_path, folder)
            if os.path.isdir(folder_path):
                volume_folders.append(folder)
        
        logger.info(f"  Found {len(volume_folders)} volume folders: {sorted(volume_folders)}")
        
        # Process each volume folder
        for volume_folder in volume_folders:
            volume_path = os.path.join(material_path, volume_folder)
            
            contcar_path = os.path.join(volume_path, "CONTCAR")
            oszicar_path = os.path.join(volume_path, "OSZICAR")
            
            if not (os.path.exists(contcar_path) and os.path.exists(oszicar_path)):
                logger.warning(f"  Skipping {volume_folder}: missing CONTCAR or OSZICAR")
                continue
            
            try:
                # Read structure and calculate volume
                atoms = read(contcar_path)
                volume = atoms.get_volume()
                
                # Read energy from OSZICAR
                energy = _read_energy_from_oszicar(oszicar_path)
                
                # Store the data
                volume_energy_pairs.append({
                    "folder": volume_folder,
                    "volume": volume,
                    "energy": energy,
                    "atoms": atoms,
                    "n_atoms": len(atoms)
                })
                
                logger.info(f"    {volume_folder}: V={volume:.2f} Å³, E={energy:.4f} eV")
                
            except Exception as e:
                logger.error(f"  Error processing {volume_folder}: {str(e)}")
                continue
        
        if not volume_energy_pairs:
            logger.warning(f"No valid data found for material {material_dir}")
            continue
        
        # Sort by volume
        volume_energy_pairs.sort(key=lambda x: x["volume"])
        
        # Find minimum energy for reference
        energies = [pair["energy"] for pair in volume_energy_pairs]
        min_energy = min(energies)
        min_energy_idx = energies.index(min_energy)
        equilibrium_volume = volume_energy_pairs[min_energy_idx]["volume"]
        
        # Store in eos_data
        eos_data[material_dir] = {
            "points": volume_energy_pairs,
            "min_energy": min_energy,
            "equilibrium_volume": equilibrium_volume,
            "n_points": len(volume_energy_pairs),
            "source": "vasp"
        }
        
        logger.info(f"  Successfully processed {len(volume_energy_pairs)} points for {material_dir}")
        logger.info(f"  Equilibrium: V={equilibrium_volume:.2f} Å³, E={min_energy:.4f} eV")
    
    if not eos_data:
        raise ValueError(f"No valid EOS data found in dataset {dataset_name}")
    
    # Convert Atoms objects to JSON format before saving
    eos_data_for_json = {}
    for material, material_data in eos_data.items():
        eos_data_for_json[material] = {
            "min_energy": material_data["min_energy"],
            "equilibrium_volume": material_data["equilibrium_volume"],
            "n_points": material_data["n_points"],
            "source": material_data["source"],
            "points": []
        }
        
        for point in material_data["points"]:
            # Convert Atoms to JSON string
            atoms_obj = point["atoms"]
            buffer = io.StringIO()
            write(buffer, atoms_obj, format='json')
            atoms_json_str = buffer.getvalue()
            
            point_data = {
                "folder": point["folder"],
                "volume": point["volume"],
                "energy": point["energy"],
                "n_atoms": point["n_atoms"],
                "atoms_json": atoms_json_str
            }
            eos_data_for_json[material]["points"].append(point_data)
    
    # Save processed data
    output_path = os.path.join(save_directory, f"{dataset_basename}_eos.json")
    
    with open(output_path, 'w') as f:
        json.dump(eos_data_for_json, f, indent=2)
    
    logger.info(f"Successfully processed {len(eos_data)} materials")
    logger.info(f"EOS data saved to: {output_path}")
    
    # Print summary
    print(f"\nEOS Data Preprocessing Summary:")
    print(f"Dataset: {dataset_name}")
    print(f"Materials processed: {len(eos_data)}")
    for material, data in eos_data.items():
        print(f"  - {material}: {data['n_points']} points, Eq. volume: {data['equilibrium_volume']:.2f} Å³")
    print(f"Output saved to: {output_path}")
    
    return output_path


def _read_energy_from_oszicar(oszicar_path):
    """Read final energy from OSZICAR file."""
    with open(oszicar_path, 'r') as f:
        lines = f.readlines()
    
    # Find the last line with E0
    for line in reversed(lines):
        if 'E0=' in line:
            # E0= value is after 'E0='
            parts = line.split('E0=')[1].split()
            return float(parts[0])
    
    # Alternative format: last line with numerical values
    for line in reversed(lines):
        parts = line.split()
        if len(parts) > 2:
            try:
                # Try to parse the 3rd column (usually energy)
                return float(parts[2])
            except ValueError:
                continue
    
    raise ValueError(f"Could not find energy in {oszicar_path}")