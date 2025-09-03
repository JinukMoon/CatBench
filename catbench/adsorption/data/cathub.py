"""
CatHub data processing for CatBench.

This module provides functions for downloading and processing catalysis reaction data 
from the CatHub database.
"""

import os
import time
import copy
import yaml
import json
import traceback
import requests
import io
import logging
from ase.io import read
from catbench.utils.io_utils import get_raw_data_directory, get_raw_data_path, save_json

GRAPHQL = "http://api.catalysis-hub.org/graphql"


def fetch(query):
    """Fetch data from CatHub GraphQL API."""
    return requests.get(GRAPHQL, {"query": query}).json()["data"]


def reactions_from_dataset(pub_id, page_size=40, logger=None):
    """
    Download reactions from CatHub dataset.
    
    Args:
        pub_id: Publication ID or dataset tag
        page_size: Number of reactions per page
        logger: Logger instance for progress tracking
        
    Returns:
        List of reaction data
    """
    reactions = []
    has_next_page = True
    start_cursor = ""
    page = 0
    while has_next_page:
        data = fetch(
            f"""{{
      reactions(pubId: "{pub_id}", first: {page_size}, after: "{start_cursor}") {{
        totalCount
        pageInfo {{
          hasNextPage
          hasPreviousPage
          startCursor
          endCursor
        }}
        edges {{
          node {{
            Equation
            reactants
            products
            reactionEnergy
            reactionSystems {{
              name
              systems {{
                energy
                InputFile(format: "json")
              }}
            }}
          }}
        }}
      }}
    }}"""
        )
        has_next_page = data["reactions"]["pageInfo"]["hasNextPage"]
        start_cursor = data["reactions"]["pageInfo"]["endCursor"]
        page += 1
        
        # Log download progress
        total_count = data["reactions"]["totalCount"]
        downloaded_so_far = page_size * page if page_size * page < total_count else total_count
        
        if logger:
            logger.info(f"Downloaded {downloaded_so_far}/{total_count} reactions for {pub_id} "
                       f"({downloaded_so_far/total_count*100:.1f}% complete)")
        
        reactions.extend(map(lambda x: x["node"], data["reactions"]["edges"]))

    return reactions


def aseify_reactions(reactions):
    """
    Convert reaction data to ASE atoms objects.
    
    Args:
        reactions: List of reaction data from CatHub
    """
    for i, reaction in enumerate(reactions):
        for j, _ in enumerate(reactions[i]["reactionSystems"]):
            system_info = reactions[i]["reactionSystems"][j].pop("systems")

            with io.StringIO() as tmp_file:
                tmp_file.write(system_info.pop("InputFile"))
                tmp_file.seek(0)
                atoms = read(tmp_file, format="json")
                atoms.pbc = True
                reactions[i]["reactionSystems"][j]["atoms"] = atoms

            reactions[i]["reactionSystems"][j]["energy"] = system_info["energy"]

        reactions[i]["reactionSystems"] = {
            x["name"]: {"atoms": x["atoms"], "energy": x["energy"]}
            for x in reactions[i]["reactionSystems"]
        }


def cathub_preprocessing(benchmark, adsorbate_integration=None):
    """
    Download and preprocess CatHub data for MLIP benchmarking.
    
    This function downloads catalysis reaction data from the CatHub database,
    processes the structures and energies, and converts them into a standardized
    format suitable for MLIP benchmarking calculations.
    
    Input Requirements:
        - Internet connection for downloading from CatHub API
        - Valid CatHub publication IDs or dataset tags
        
    Output Files:
        - JSON files: {benchmark}.json containing raw downloaded data
        - JSON files: {benchmark}_adsorption.json with processed benchmark data  
        - YAML files: {output_name}.yml with metadata (for multiple benchmarks)
        - LOG files: cathub_preprocessing.log with processing details
        
    Data Processing Steps:
        1. Download reaction data from CatHub GraphQL API
        2. Convert InputFile JSON strings to ASE Atoms objects
        3. Validate reaction stoichiometry and energy consistency
        4. Filter out incomplete or invalid reactions
        5. Apply adsorbate name integration if specified
        6. Save processed data in standardized JSON format
    
    Args:
        benchmark (str or list): Single benchmark tag or list of benchmark tags.
                               Examples: "AraComputational2022", 
                                       ["AraComputational2022", "AlonsoStrain2023"]
        adsorbate_integration (dict, optional): Mapping for adsorbate name unification.
                                              Format: {"source_name": "target_name"}
                                              Example: {"OH2": "H2O", "H2O2": "OOH"}
                                              
    Raises:
        ValueError: If reaction energy validation fails
        KeyError: If required reaction systems are missing
        ConnectionError: If CatHub API is not accessible
        
    Note:
        The function automatically handles duplicate reaction names and validates
        reaction stoichiometry. Invalid reactions are filtered out with error messages.
    """
    save_directory = get_raw_data_directory()
    os.makedirs(save_directory, exist_ok=True)
    
    # Convert single string to list for uniform processing
    benchmarks = [benchmark] if isinstance(benchmark, str) else benchmark
    
    # Check if any downloads are needed and setup logging if so
    download_needed = any(not os.path.exists(os.path.join(save_directory, f"{bench}.json")) 
                         for bench in benchmarks)
    
    logger = None
    if download_needed:
        log_name = benchmark if isinstance(benchmark, str) else "multiple_tag"
        log_file = os.path.join(save_directory, f"{log_name}_preprocessing.log")
        logger = logging.getLogger(f"catbench_{log_name}")
        logger.setLevel(logging.INFO)
        # Remove existing handlers to prevent duplication
        if logger.hasHandlers():
            logger.handlers.clear()
        handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        logger.info(f"Starting CatHub data download for benchmark: {benchmark}")
    
    # Initialize combined data structure
    combined_reactions = []
    
    for bench in benchmarks:
        path_json = os.path.join(save_directory, f"{bench}.json")
        # Create separate logger/handler for each benchmark
        bench_logger = None
        log_file = os.path.join(save_directory, f"{bench}_preprocessing.log")
        bench_logger = logging.getLogger(f"catbench_{bench}")
        bench_logger.setLevel(logging.INFO)
        if bench_logger.hasHandlers():
            bench_logger.handlers.clear()
        handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        bench_logger.addHandler(handler)
        bench_logger.propagate = False
        bench_logger.info(f"Starting CatHub data download for benchmark: {bench}")
        
        # Get reactions for benchmark (preliminary download)
        if not os.path.exists(path_json):
            bench_logger.info(f"Downloading reactions for benchmark: {bench}")
            raw_reactions = reactions_from_dataset(bench, logger=bench_logger)
            bench_logger.info(f"Download completed for {bench}: {len(raw_reactions)} reactions")
            raw_reactions_json = {"raw_reactions": raw_reactions}
            save_json(raw_reactions_json, path_json, use_numpy_encoder=False)
            bench_logger.info(f"Saved raw data to {path_json}")
        else:
            with open(path_json, "r") as file:
                raw_reactions_json = json.load(file)
        combined_reactions.extend(raw_reactions_json["raw_reactions"])
                    # Remove handlers for each benchmark to prevent memory leaks
        bench_logger.handlers.clear()
    
    # Generate output filename based on input type
    if isinstance(benchmark, str):
        output_name = benchmark
    else:
        output_name = "multiple_tag"
        # Save benchmark information to yaml file
        benchmark_info = {
            "benchmarks": sorted(benchmarks),
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_reactions": len(combined_reactions)
        }
        yaml_path = os.path.join(save_directory, f"{output_name}.yml")
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(benchmark_info, yaml_file, default_flow_style=False)
    
    # JSON format with _catbench suffix
    path_output = get_raw_data_path(output_name)
    
    if not os.path.exists(path_output):
        # Setup logging if not already done (e.g., no download was needed)
        if logger is None:
            log_file = os.path.join(save_directory, f"{output_name}_preprocessing.log")
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, mode='w', encoding='utf-8')
                    # Console output removed for cleaner debugging
                ]
            )
            logger = logging.getLogger(__name__)
            logger.info(f"Starting CatHub preprocessing for benchmark: {benchmark}")
        
        # Continue with processing logs
        logger.info(f"Starting data processing for benchmark: {benchmark}")
        if adsorbate_integration:
            logger.info(f"Adsorbate integration mapping: {adsorbate_integration}")
        
        if isinstance(benchmark, str):
            logger.info(f"Processing single benchmark: {benchmark}")
        else:
            logger.info(f"Processing multiple benchmarks: {benchmarks}")
            logger.info(f"Benchmark metadata saved to {os.path.join(save_directory, f'{output_name}.yml')}")
        
        logger.info(f"Total combined reactions to process: {len(combined_reactions)}")
        logger.info(f"Data processing output will be saved to {path_output}")
        
        # Process combined reactions
        dat = copy.deepcopy(combined_reactions)
        logger.info("Converting reaction data to ASE atoms objects...")
        aseify_reactions(dat)

        data_total = {}
        tags = []
        
        logger.info(f"Processing {len(dat)} reactions for energy validation...")

        for i, _ in enumerate(dat):
            try:
                input = {}
                reactants_json = dat[i]["reactants"]
                reactants_dict = json.loads(reactants_json)

                products_json = dat[i]["products"]
                products_dict = json.loads(products_json)



                # Generate tag safely - use 'star' if available, otherwise use first available system
                if "star" in dat[i]["reactionSystems"]:
                    sym = dat[i]["reactionSystems"]["star"]["atoms"].get_chemical_formula()
                else:
                    # Use first available system for tag generation
                    first_key = next(iter(dat[i]["reactionSystems"]))
                    sym = dat[i]["reactionSystems"][first_key]["atoms"].get_chemical_formula()
                
                reaction_name = dat[i]["Equation"]
                tag = sym + "_" + reaction_name
                if tag in tags:
                    count = tags.count(tag)
                    tags.append(tag)
                    tag = f"{tag}_{count}"
                else:
                    tags.append(tag)



                # Step 1: Add all structures from reactants and products dictionaries as-is
                for key in dat[i]["reactionSystems"]:
                    if key in reactants_dict:
                        input[key] = {
                            "stoi": -reactants_dict[key],
                            "atoms": dat[i]["reactionSystems"][key]["atoms"],
                            "energy_ref": dat[i]["reactionSystems"][key]["energy"],
                        }
                    elif key in products_dict:
                        input[key] = {
                            "stoi": products_dict[key],
                            "atoms": dat[i]["reactionSystems"][key]["atoms"],
                            "energy_ref": dat[i]["reactionSystems"][key]["energy"],
                        }

                # Filter for single adsorbate reactions only
                star_structures = [key for key in input.keys() if "star" in key]
                if len(star_structures) != 2:
                    logger.info(f"Filtered - {tag}: Multi-adsorbate reaction (found {len(star_structures) - 1} adslab structures, expected 1)")
                    continue

                # Step 2: First energy validation with original coefficients
                energy_check = 0
                for structure in input:
                    energy_check += (
                        input[structure]["energy_ref"] * input[structure]["stoi"]
                    )
                
                validation_passed = False
                if abs(dat[i]["reactionEnergy"] - energy_check) <= 0.001:
                    # Validation passed with original coefficients
                    validation_passed = True
                    logger.debug(f"Energy validation passed for {tag} with original coefficients")
                else:
                    # Step 3: Try with adslab coefficient = 1
                    logger.debug(f"First validation failed for {tag}, trying with adslab coefficient = 1")
                    
                    # Find the adslab structure (not "star")
                    adslab_key = None
                    for key in input.keys():
                        if "star" in key and key != "star":
                            adslab_key = key
                            break
                    
                    if adslab_key:
                        original_coeff = input[adslab_key]["stoi"]
                        input[adslab_key]["stoi"] = 1  # Try with coefficient 1
                        
                        # Recalculate energy with corrected coefficient
                        energy_check_retry = 0
                        for structure in input:
                            energy_check_retry += (
                                input[structure]["energy_ref"] * input[structure]["stoi"]
                            )
                        
                        if abs(dat[i]["reactionEnergy"] - energy_check_retry) <= 0.001:
                            # Validation passed with coefficient = 1
                            validation_passed = True
                            logger.info(f"Energy validation passed for {tag} with adslab coefficient changed from {original_coeff} to 1")
                        else:
                            # Revert to original coefficient if still fails
                            input[adslab_key]["stoi"] = original_coeff
                
                # Step 4: Save or filter based on validation result
                if validation_passed:
                    data_total[tag] = {}
                    data_total[tag]["raw"] = input
                    data_total[tag]["ref_ads_eng"] = dat[i]["reactionEnergy"]
                else:
                    # Both validations failed
                    logger.info(f"Filtered - {tag}: Stoichiometry inconsistency even after trying coefficient adjustment")
                    logger.debug(f"  CatHub energy: {dat[i]['reactionEnergy']:.6f} eV")
                    logger.debug(f"  Original calculation: {energy_check:.6f} eV (diff: {abs(dat[i]['reactionEnergy'] - energy_check):.6f} eV)")
                    if adslab_key:
                        logger.debug(f"  With adslab coeff=1: {energy_check_retry:.6f} eV (diff: {abs(dat[i]['reactionEnergy'] - energy_check_retry):.6f} eV)")
                    
                    if tag in tags:
                        tags.remove(tag)
                    continue

                # Apply adsorbate integration if specified
                if adsorbate_integration:
                    integration_applied = False
                    for key in list(data_total[tag]["raw"].keys()):
                        if "star" in key and key != "star":
                            adsorbate = key[:-4]
                            if adsorbate in adsorbate_integration:
                                integrated_key = f"{adsorbate_integration[adsorbate]}star"
                                data_total[tag]["raw"][integrated_key] = data_total[tag]["raw"].pop(key)
                                logger.debug(f"Integrated adsorbate in {tag}: {key} → {integrated_key}")
                                integration_applied = True
                    if integration_applied:
                        logger.info(f"Applied adsorbate integration to reaction: {tag}")
                
                # Add adsorbate indices detection using common utility
                from catbench.utils.data_utils import detect_adsorbate_indices
                
                # Find slab and adslab structures
                slab_key = "star"
                adslab_key = None
                for key in data_total[tag]["raw"].keys():
                    if "star" in key and key != "star":
                        adslab_key = key
                        break
                
                if slab_key in data_total[tag]["raw"] and adslab_key:
                    slab_atoms = data_total[tag]["raw"][slab_key]["atoms"]
                    adslab_atoms = data_total[tag]["raw"][adslab_key]["atoms"]
                    
                    # Use common detection function
                    adsorbate_indices = detect_adsorbate_indices(slab_atoms, adslab_atoms)
                    
                    # Store adsorbate indices
                    data_total[tag]["adsorbate_indices"] = adsorbate_indices
                    logger.debug(f"Detected adsorbate indices for {tag}: {adsorbate_indices}")
                else:
                    # If we can't detect, set empty list
                    data_total[tag]["adsorbate_indices"] = []
                    logger.warning(f"Could not detect adsorbate indices for {tag}")
                
                # Log successful processing
                logger.debug(f"Successfully processed reaction: {tag}")
                logger.debug(f"  Reaction energy: {dat[i]['reactionEnergy']:.6f} eV")
                logger.debug(f"  Number of structures: {len(data_total[tag]['raw'])}")

            except Exception as e:
                logger.error(f"Unexpected error processing reaction {i+1}/{len(dat)}: {e}")
                logger.error(f"Reaction tag: {tag if 'tag' in locals() else 'Unknown'}")
                logger.error(f"Traceback: {traceback.format_exc()}")

        # Processing complete - show detailed statistics
        filtered_count = len(dat) - len(data_total)
        success_rate = (len(data_total) / len(dat) * 100) if len(dat) > 0 else 0
        
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total reactions attempted: {len(dat)}")
        logger.info(f"Successfully processed: {len(data_total)} ({success_rate:.1f}%)")
        logger.info(f"Filtered out: {filtered_count} ({filtered_count/len(dat)*100:.1f}%)")
        logger.info("=" * 60)
        
        # Also print to console for immediate visibility
        print("\n" + "=" * 60)
        print("CATHUB DATA PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Successfully processed: {len(data_total)}/{len(dat)} reactions ({success_rate:.1f}%)")
        print(f"Filtered out: {filtered_count} reactions")
        print(f"Output file: {path_output}")
        print("=" * 60 + "\n")
        
        logger.info(f"Saving processed data to {path_output}")
        
        # JSON format
        from catbench.utils.data_utils import save_catbench_json
        save_catbench_json(data_total, path_output)
        logger.info("Data processing and saving completed successfully!")
    else:
        print(f"Processed data already exists at {path_output}")
        print("   Skipping processing to avoid overwriting existing data.")
        print("   Delete the file if you want to reprocess the data.")


def download(benchmark_tags):
    """
    Download raw reaction data from CatHub (without processing).
    
    Args:
        benchmark_tags: Single tag or list of tags
        
    Returns:
        List of raw reaction data
    """
    if isinstance(benchmark_tags, str):
        benchmark_tags = [benchmark_tags]
    
    all_reactions = []
    for tag in benchmark_tags:
        reactions = reactions_from_dataset(tag)
        all_reactions.extend(reactions)
    
    return all_reactions 