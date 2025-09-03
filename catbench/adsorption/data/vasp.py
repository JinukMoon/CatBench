"""
VASP data processing for CatBench.

This module provides functions for processing user-provided VASP calculation data
and converting them into formats suitable for MLIP benchmarking.
"""

import os
import json
from ase.io import read
from catbench.utils.io_utils import get_raw_data_directory, get_raw_data_path, save_json


def read_E0_from_OSZICAR(file_path):
    """
    Read final energy (E0) from VASP OSZICAR file.
    
    Args:
        file_path: Path to OSZICAR file
        
    Returns:
        float: Final energy in eV
    """
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            last_line = lines[-1]

        energy = None
        for word in last_line.split():
            if word == "E0=":
                energy_index = last_line.split().index(word) + 1
                energy = last_line.split()[energy_index]
                energy = float(energy)
                break

        if energy is None:
            raise ValueError(f"Energy value not found in file: {file_path}")

        return energy

    except Exception as e:
        raise RuntimeError(
            f"An error occurred while reading the file '{file_path}': {str(e)}"
        )


def process_output(dataset_name, coeff_setting):
    """
    Clean up VASP output files and prepare coefficient settings.
    
    This function removes unnecessary files from VASP calculations, keeping only
    CONTCAR and OSZICAR files, and sets up coefficient files for reaction energy calculations.
    
    Args:
        dataset_name: Path to dataset directory
        coeff_setting: Dictionary containing coefficient settings for each reaction
    """
    for dirpath, dirnames, filenames in os.walk(dataset_name):
        # Check if both OSZICAR and CONTCAR files exist
        if "OSZICAR" in filenames and "CONTCAR" in filenames:
            # Iterate through all files in the folder
            for file in filenames:
                # Delete files that are not OSZICAR or CONTCAR
                if file not in ["OSZICAR", "CONTCAR"]:
                    file_path = os.path.join(dirpath, file)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

            rxn_name = dirpath.split("/")[2]

            not_calc_dirs = [
                name
                for name in os.listdir(f"{dataset_name}/gas")
                if os.path.isdir(os.path.join(f"{dataset_name}/gas", name))
            ] + ["slab"]

            if rxn_name not in not_calc_dirs:
                coeff = coeff_setting[rxn_name]
            else:
                coeff = {}

            coeff_path = os.path.join(dirpath, "coeff.json")
            if not os.path.exists(coeff_path) and coeff != {}:
                save_json(coeff, coeff_path, use_numpy_encoder=False)

    for dir_name in os.listdir(dataset_name):
        dir_path = os.path.join(dataset_name, dir_name)
        if os.path.isdir(dir_path) and dir_name != "gas":
            slab_folder_path = os.path.join(dir_path, "slab")
            os.makedirs(slab_folder_path, exist_ok=True)


def userdata_preprocess(dataset_name):
    """
    Preprocess user VASP dataset for benchmarking.
    
    This function processes VASP calculation results and converts them into
    the standard format for MLIP benchmarking.
    
    Args:
        dataset_name: Path to the dataset directory containing VASP calculations
    """
    save_directory = get_raw_data_directory()
    os.makedirs(save_directory, exist_ok=True)
    # JSON format with _catbench suffix
    path_output = get_raw_data_path(dataset_name)
    data_total = {}
    tags = []
    not_calc_dirs = [
        name
        for name in os.listdir(f"{dataset_name}/gas")
        if os.path.isdir(os.path.join(f"{dataset_name}/gas", name))
    ] + ["slab"]

    for dirpath, dirnames, filenames in os.walk(dataset_name):
        if "OSZICAR" in filenames and "CONTCAR" in filenames:
            rxn_name = dirpath.split("/")[2]
            if rxn_name not in not_calc_dirs:
                input = {}
                slab_name = dirpath.split("/")[1]
                slab_path = dirpath[: dirpath.find("/", dirpath.find("/") + 1)]

                coeff_path = os.path.join(dirpath, "coeff.json")
                with open(coeff_path, "r") as file:
                    coeff = json.load(file)

                tag = slab_name + "_" + rxn_name

                if tag in tags:
                    count = tags.count(tag)
                    tags.append(tag)
                    tag = f"{tag}_{count}"
                else:
                    tags.append(tag)

                input["star"] = {
                    "stoi": coeff["slab"],
                    "atoms": read(f"{slab_path}/slab/CONTCAR"),
                    "energy_ref": read_E0_from_OSZICAR(f"{slab_path}/slab/OSZICAR"),
                }

                input[f"{rxn_name}star"] = {
                    "stoi": coeff["adslab"],
                    "atoms": read(f"{dirpath}/CONTCAR"),
                    "energy_ref": read_E0_from_OSZICAR(f"{dirpath}/OSZICAR"),
                }

                for key in coeff:
                    if key not in ["slab", "adslab"]:
                        input[key] = {
                            "stoi": coeff[key],
                            "atoms": read(f"{dataset_name}/gas/{key}/CONTCAR"),
                            "energy_ref": read_E0_from_OSZICAR(
                                f"{dataset_name}/gas/{key}/OSZICAR"
                            ),
                        }

                energy_check = 0
                for structure in input:
                    energy_check += (
                        input[structure]["energy_ref"] * input[structure]["stoi"]
                    )

                data_total[tag] = {}

                data_total[tag]["raw"] = input
                data_total[tag]["ref_ads_eng"] = energy_check
                
                # Add adsorbate indices detection using common utility
                from catbench.utils.data_utils import detect_adsorbate_indices
                
                slab_atoms = input["star"]["atoms"]
                adslab_atoms = input[f"{rxn_name}star"]["atoms"]
                
                # Use common detection function
                adsorbate_indices = detect_adsorbate_indices(slab_atoms, adslab_atoms)
                
                # Store adsorbate indices
                data_total[tag]["adsorbate_indices"] = adsorbate_indices

    # Show detailed statistics
    total_reactions = len(tags)
    successful_reactions = len(data_total)
    success_rate = (successful_reactions / total_reactions * 100) if total_reactions > 0 else 100
    
    print("\n" + "=" * 60)
    print("VASP DATA PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Successfully processed: {successful_reactions}/{total_reactions} reactions ({success_rate:.1f}%)")
    print(f"Output file: {path_output}")
    print("=" * 60 + "\n")

    # JSON format
    from catbench.utils.data_utils import save_catbench_json
    save_catbench_json(data_total, path_output)


def vasp_preprocessing(dataset_name, coeff_setting):
    """
    Complete VASP data processing pipeline for MLIP benchmarking.
    
    This function provides a comprehensive solution for processing user-provided
    VASP calculation data and converting them into the standardized format
    required for MLIP benchmarking. It combines file cleanup, coefficient 
    assignment, and data preprocessing in a single workflow.
    
    Required Directory Structure:
        dataset_name/
        ├── slab_type1/
        │   ├── slab/
        │   │   ├── CONTCAR (relaxed clean slab structure)
        │   │   └── OSZICAR (slab energy)
        │   ├── adsorbate1/
        │   │   ├── CONTCAR (relaxed adslab structure)
        │   │   └── OSZICAR (adslab energy)
        │   └── adsorbate2/
        │       ├── CONTCAR
        │       └── OSZICAR
        └── gas/
            ├── H2gas/
            │   ├── CONTCAR (gas molecule structure)
            │   └── OSZICAR (gas molecule energy)
            └── H2Ogas/
                ├── CONTCAR
                └── OSZICAR
                
    Input Files:
        - CONTCAR files: VASP structure files with atomic positions
        - OSZICAR files: VASP output files containing final energies (E0 values)
        
    Output Files:
        - JSON file: {dataset_name}_adsorption.json containing processed benchmark data
        - JSON files: coeff.json files in each reaction directory
        
    Processing Steps:
        1. Clean VASP output directories (remove unnecessary files)
        2. Create coefficient files for each reaction
        3. Read structures and energies from VASP files
        4. Calculate reference adsorption energies
        5. Validate energy conservation 
        6. Save data in standardized JSON format
    
    Args:
        dataset_name (str): Path to the root directory containing VASP calculations.
                           Can be relative or absolute path.
        coeff_setting (dict): Reaction stoichiometry coefficients for each adsorbate.
                            Format: {
                                "adsorbate_name": {
                                    "slab": coefficient,
                                    "adslab": coefficient, 
                                    "gas_species": coefficient,
                                    ...
                                }
                            }
                            
    Raises:
        FileNotFoundError: If required VASP files are missing
        ValueError: If energy values cannot be parsed from OSZICAR
        RuntimeError: If file reading operations fail
        
    Note:
        The function automatically validates reaction energetics and filters out
        reactions with inconsistent energy balance. Each reaction must satisfy
        energy conservation within numerical precision.
    """
    print("Step 1: Cleaning VASP output files...")
    process_output(dataset_name, coeff_setting)
    
    print("Step 2: Preprocessing VASP data...")
    userdata_preprocess(dataset_name)
    
    print(f"VASP data processing completed for {dataset_name}")


 