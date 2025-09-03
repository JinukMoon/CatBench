"""
CatHub Relative Energy Data Preprocessing for CatBench.

This module provides functions for processing CatHub data specifically for
relative energy benchmarking tasks, focusing on surface energy calculations.
"""

import os
import json
import copy
import logging
import time
from collections import Counter
from ase import Atoms
from ase.io import read

from catbench.adsorption.data.cathub import reactions_from_dataset, aseify_reactions
from catbench.utils.data_utils import save_catbench_json


def surface_energy_cathub_preprocessing(benchmark, save_directory="raw_data"):
    """
    Download and preprocess CatHub data for surface energy benchmarking.
    
    This function downloads catalysis reaction data from the CatHub database,
    identifies surface-bulk pairs, and processes them into standardized format
    for surface energy calculations.
    
    Processing Steps:
        1. Download reaction data from CatHub (or use existing JSON)
        2. Identify reactions containing both 'star' (surface) and 'bulk' systems
        3. Analyze surface and bulk compositions to find integer multiples
        4. Calculate normalization factors and validate stoichiometry
        5. Remove duplicates based on composition pairs and energy consistency
        6. Save processed data in standardized pickle format
    
    Args:
        benchmark (str): CatHub benchmark tag (e.g., "AraComputational2022")
        save_directory (str, optional): Directory to save output files. Default: "raw_data"
        
    Output Files:
        - {benchmark}_surface_energy.json: Processed surface energy data
        - {benchmark}_surface_energy_preprocessing.log: Processing details
        
    Returns:
        str: Path to the created JSON file
        
    Raises:
        ValueError: If surface and bulk compositions are not integer multiples
        FileNotFoundError: If CatHub data cannot be downloaded or found
    """
    
    # Setup logging
    os.makedirs(save_directory, exist_ok=True)
    log_file = os.path.join(save_directory, f"{benchmark}_surface_energy_preprocessing.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting CatHub surface energy preprocessing for benchmark: {benchmark}")
    
    # Download CatHub data if not exists
    json_path = os.path.join(save_directory, f"{benchmark}.json")
    if not os.path.exists(json_path):
        logger.info("Downloading CatHub data...")
        raw_reactions = reactions_from_dataset(benchmark, logger=logger)
        # Save to JSON in the same format as cathub_preprocessing
        raw_reactions_json = {"raw_reactions": raw_reactions}
        with open(json_path, 'w') as f:
            json.dump(raw_reactions_json, f, indent=4)
        logger.info(f"Downloaded data saved to: {json_path}")
    else:
        logger.info(f"Using existing CatHub data: {json_path}")
    
    # Load reactions from JSON (both new and existing files use same format)
    with open(json_path, 'r') as f:
        raw_reactions_json = json.load(f)
    combined_reactions = raw_reactions_json["raw_reactions"]
    
    logger.info(f"Processing {len(combined_reactions)} reactions for surface energy data...")
    
    # Process reactions
    dat = copy.deepcopy(combined_reactions)
    aseify_reactions(dat)
    
    surface_energy_data = {}
    processed_pairs = []  # Track processed composition pairs
    
    # Statistics tracking
    total_reactions = len(dat)
    skipped_no_bulk = 0
    skipped_missing_energy = 0
    skipped_composition_mismatch = 0
    skipped_duplicates = 0
    skipped_errors = 0
    successfully_added = 0
    
    logger.info(f"Starting surface energy data extraction from {total_reactions} reactions...")
    
    for i, reaction in enumerate(dat):
        try:
            reaction_equation = reaction.get("Equation", f"Reaction_{i}")
            logger.debug(f"Processing reaction {i+1}/{total_reactions}: {reaction_equation}")
            
            reaction_systems = reaction["reactionSystems"]
            
            # Check if both 'star' and 'bulk' exist
            if "star" not in reaction_systems or "bulk" not in reaction_systems:
                logger.debug(f"Skipping reaction {i+1}: missing star or bulk system")
                logger.debug(f"  Available systems: {list(reaction_systems.keys())}")
                skipped_no_bulk += 1
                continue
                
            star_atoms = reaction_systems["star"]["atoms"]
            bulk_atoms = reaction_systems["bulk"]["atoms"]
            star_energy = reaction_systems["star"]["energy"]
            bulk_energy = reaction_systems["bulk"]["energy"]
            
            # Validate energy values
            if star_energy is None or bulk_energy is None:
                logger.debug(f"Skipping reaction {i+1}: missing energy values")
                logger.debug(f"  Star energy: {star_energy}, Bulk energy: {bulk_energy}")
                skipped_missing_energy += 1
                continue
            
            # Analyze compositions
            star_composition = _get_composition(star_atoms)
            bulk_composition = _get_composition(bulk_atoms)
            
            logger.debug(f"Reaction {i+1}: Star composition {star_composition}, Bulk composition {bulk_composition}")
            
            # Find integer multiple relationship
            n_factor = _find_composition_multiple(star_composition, bulk_composition)
            
            if n_factor is None:
                logger.warning(f"Skipping reaction {i+1}: compositions are not integer multiples")
                logger.warning(f"  Star: {star_composition}, Bulk: {bulk_composition}")
                skipped_composition_mismatch += 1
                continue
                
            logger.debug(f"Reaction {i+1}: Found valid n-factor = {n_factor}")
            
            # Check for duplicates
            pair_signature = (tuple(sorted(star_composition.items())), 
                            tuple(sorted(bulk_composition.items())))
            
            is_duplicate = False
            for processed_pair in processed_pairs:
                if (processed_pair["signature"] == pair_signature and
                    abs(processed_pair["star_energy"] - star_energy) < 0.001 and
                    abs(processed_pair["bulk_energy"] - bulk_energy) < 0.001):
                    logger.debug(f"Skipping reaction {i+1}: duplicate surface energy calculation")
                    logger.debug(f"  Same composition pair and energies already processed")
                    is_duplicate = True
                    break
                    
            if is_duplicate:
                skipped_duplicates += 1
                continue
                
            # Generate unique tag
            formula = _composition_to_formula(star_composition)
            tag = f"{formula}_surface"
            counter = 1
            original_tag = tag
            while tag in surface_energy_data:
                tag = f"{original_tag}_{counter}"
                counter += 1
                
            # Store surface energy data
            surface_energy_data[tag] = {
                "star": {
                    "atoms": star_atoms,
                    "energy": star_energy,
                    "composition": star_composition
                },
                "bulk": {
                    "atoms": bulk_atoms, 
                    "energy": bulk_energy,
                    "composition": bulk_composition
                },
                "n_factor": n_factor,
                "reaction_equation": reaction_equation,
                "source": "cathub"
            }
            
            # Track processed pair
            processed_pairs.append({
                "signature": pair_signature,
                "star_energy": star_energy,
                "bulk_energy": bulk_energy
            })
            
            successfully_added += 1
            logger.info(f"Successfully added surface energy system: {tag}")
            logger.debug(f"  Star energy: {star_energy:.6f} eV, Bulk energy: {bulk_energy:.6f} eV, n-factor: {n_factor}")
            
        except Exception as e:
            logger.error(f"Error processing reaction {i+1}: {str(e)}")
            logger.error(f"  Reaction equation: {reaction.get('Equation', 'Unknown')}")
            skipped_errors += 1
            continue
    
    # Log processing statistics
    logger.info("Surface energy data extraction completed!")
    logger.info(f"Processing statistics:")
    logger.info(f"  Total reactions processed: {total_reactions}")
    logger.info(f"  Successfully added: {successfully_added}")
    logger.info(f"  Skipped - missing bulk system: {skipped_no_bulk}")
    logger.info(f"  Skipped - missing energy data: {skipped_missing_energy}")
    logger.info(f"  Skipped - composition mismatch: {skipped_composition_mismatch}")
    logger.info(f"  Skipped - duplicate systems: {skipped_duplicates}")
    logger.info(f"  Skipped - processing errors: {skipped_errors}")
    logger.info(f"  Success rate: {successfully_added/total_reactions*100:.1f}%")
    
    if not surface_energy_data:
        raise ValueError(f"No valid surface energy systems found in benchmark {benchmark}")
        
    # Save processed data
    output_path = os.path.join(save_directory, f"{benchmark}_surface_energy.json")
    save_catbench_json(surface_energy_data, output_path)
        
    logger.info(f"Successfully processed {len(surface_energy_data)} surface energy systems")
    logger.info(f"Surface energy data saved to: {output_path}")
    
    return output_path


def _get_composition(atoms):
    """Extract composition dictionary from ASE atoms object."""
    symbols = atoms.get_chemical_symbols()
    return dict(Counter(symbols))


def _composition_to_formula(composition):
    """Convert composition dict to chemical formula string."""
    formula_parts = []
    for element in sorted(composition.keys()):
        count = composition[element]
        if count == 1:
            formula_parts.append(element)
        else:
            formula_parts.append(f"{element}{count}")
    return "".join(formula_parts)


def _find_composition_multiple(star_comp, bulk_comp):
    """
    Find integer multiple n such that star_comp = n * bulk_comp.
    
    Returns:
        int or None: The multiple factor if found, None otherwise
    """
    if not bulk_comp:
        return None
        
    # Check if all elements in bulk are present in star
    for element in bulk_comp:
        if element not in star_comp:
            return None
            
    # Calculate potential n factors for each element
    n_factors = []
    for element in bulk_comp:
        if bulk_comp[element] == 0:
            continue
        n = star_comp[element] / bulk_comp[element]
        if n != int(n) or n <= 0:
            return None
        n_factors.append(int(n))
        
    # Check if all n factors are the same
    if len(set(n_factors)) != 1:
        return None
        
    n = n_factors[0]
    
    # Verify that star_comp = n * bulk_comp for all elements
    for element in star_comp:
        expected_count = n * bulk_comp.get(element, 0)
        if star_comp[element] != expected_count:
            return None
            
    return n 