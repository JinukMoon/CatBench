"""
Data processing utilities for CatBench.

This module provides common data processing functions used across different
data sources (CatHub, VASP, etc.).
"""

from collections import Counter
from typing import List, Set, Dict, Any
import numpy as np
import json
import io
from ase import Atoms
from ase.io import write, read


def detect_adsorbate_indices(slab_atoms, adslab_atoms) -> List[int]:
    """
    Detect adsorbate atom indices by comparing slab and adslab structures.
    
    Uses element-based detection with z-coordinate prioritization:
    - Additional elements not in slab are all considered adsorbate
    - For existing elements with increased count, selects atoms with highest z-coordinates
    
    Args:
        slab_atoms: ASE Atoms object for clean slab
        adslab_atoms: ASE Atoms object for slab with adsorbate
        
    Returns:
        List[int]: Sorted list of adsorbate atom indices
    """
    # Element-based detection with z-coordinate prioritization
    slab_elements = Counter(slab_atoms.get_chemical_symbols())
    adslab_elements = Counter(adslab_atoms.get_chemical_symbols())
    
    adsorbate_indices_set = set()
    
    for element, count in adslab_elements.items():
        if element not in slab_elements:
            # This element doesn't exist in slab at all - must be adsorbate
            for i, atom in enumerate(adslab_atoms):
                if atom.symbol == element:
                    adsorbate_indices_set.add(i)
        elif count > slab_elements[element]:
            # This element has more atoms in adslab - select by z-coordinate
            n_extra = count - slab_elements[element]
            
            # Get all atoms of this element with their indices and z-coordinates
            element_atoms_with_z = [(i, adslab_atoms[i].position[2]) 
                                   for i, atom in enumerate(adslab_atoms) 
                                   if atom.symbol == element]
            
            # Sort by z-coordinate (highest first)
            element_atoms_with_z.sort(key=lambda x: x[1], reverse=True)
            
            # Select the top n_extra atoms by z-coordinate as adsorbate
            top_n_indices = [idx for idx, z in element_atoms_with_z[:n_extra]]
            adsorbate_indices_set.update(top_n_indices)
    
    # Return sorted list of indices
    adsorbate_indices = sorted(list(adsorbate_indices_set))
    
    return adsorbate_indices


def save_catbench_json(data_dict: Dict[str, Any], filepath: str) -> None:
    """
    Save CatBench data dictionary to JSON format.
    Converts ASE Atoms objects to JSON-serializable format.
    
    Args:
        data_dict: Dictionary containing reaction data with ASE Atoms objects
        filepath: Path to save JSON file
    """
    # Deep copy to avoid modifying original
    import copy
    json_data = copy.deepcopy(data_dict)
    
    # Convert all Atoms objects to JSON strings
    for reaction_key in json_data:
        # Handle adsorption data format (with "raw")
        if "raw" in json_data[reaction_key]:
            for structure_key in json_data[reaction_key]["raw"]:
                if "atoms" in json_data[reaction_key]["raw"][structure_key]:
                    atoms_obj = json_data[reaction_key]["raw"][structure_key]["atoms"]
                    
                    # Convert Atoms to JSON string using ASE's JSON writer
                    buffer = io.StringIO()
                    write(buffer, atoms_obj, format='json')
                    atoms_json_str = buffer.getvalue()
                    
                    # Replace atoms object with JSON string
                    json_data[reaction_key]["raw"][structure_key]["atoms_json"] = atoms_json_str
                    del json_data[reaction_key]["raw"][structure_key]["atoms"]
        
        # Handle surface energy / bulk formation data format
        # Check for "surface", "star", "bulk", "target", "references" keys
        for key in ["surface", "star", "bulk", "target"]:
            if key in json_data[reaction_key] and "atoms" in json_data[reaction_key][key]:
                atoms_obj = json_data[reaction_key][key]["atoms"]
                
                buffer = io.StringIO()
                write(buffer, atoms_obj, format='json')
                atoms_json_str = buffer.getvalue()
                
                json_data[reaction_key][key]["atoms_json"] = atoms_json_str
                del json_data[reaction_key][key]["atoms"]
        
        # Handle references in bulk formation / custom data
        if "references" in json_data[reaction_key]:
            for ref_key in json_data[reaction_key]["references"]:
                if "atoms" in json_data[reaction_key]["references"][ref_key]:
                    atoms_obj = json_data[reaction_key]["references"][ref_key]["atoms"]
                    
                    buffer = io.StringIO()
                    write(buffer, atoms_obj, format='json')
                    atoms_json_str = buffer.getvalue()
                    
                    json_data[reaction_key]["references"][ref_key]["atoms_json"] = atoms_json_str
                    del json_data[reaction_key]["references"][ref_key]["atoms"]
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Data saved to {filepath}")


def load_catbench_json(filepath: str) -> Dict[str, Any]:
    """
    Load CatBench data from JSON format.
    Converts JSON-serialized Atoms back to ASE Atoms objects.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with ASE Atoms objects restored
    """
    # Load JSON data
    with open(filepath, 'r') as f:
        json_data = json.load(f)
    
    # Convert JSON strings back to Atoms objects
    for reaction_key in json_data:
        # Handle adsorption data format (with "raw")
        if "raw" in json_data[reaction_key]:
            for structure_key in json_data[reaction_key]["raw"]:
                if "atoms_json" in json_data[reaction_key]["raw"][structure_key]:
                    atoms_json_str = json_data[reaction_key]["raw"][structure_key]["atoms_json"]
                    
                    # Convert JSON string back to Atoms object
                    buffer = io.StringIO(atoms_json_str)
                    atoms_obj = read(buffer, format='json')
                    
                    # Replace JSON string with Atoms object
                    json_data[reaction_key]["raw"][structure_key]["atoms"] = atoms_obj
                    del json_data[reaction_key]["raw"][structure_key]["atoms_json"]
        
        # Handle surface energy / bulk formation data format
        for key in ["surface", "star", "bulk", "target"]:
            if key in json_data[reaction_key] and "atoms_json" in json_data[reaction_key][key]:
                atoms_json_str = json_data[reaction_key][key]["atoms_json"]
                
                buffer = io.StringIO(atoms_json_str)
                atoms_obj = read(buffer, format='json')
                
                json_data[reaction_key][key]["atoms"] = atoms_obj
                del json_data[reaction_key][key]["atoms_json"]
        
        # Handle references in bulk formation / custom data
        if "references" in json_data[reaction_key]:
            for ref_key in json_data[reaction_key]["references"]:
                if "atoms_json" in json_data[reaction_key]["references"][ref_key]:
                    atoms_json_str = json_data[reaction_key]["references"][ref_key]["atoms_json"]
                    
                    buffer = io.StringIO(atoms_json_str)
                    atoms_obj = read(buffer, format='json')
                    
                    json_data[reaction_key]["references"][ref_key]["atoms"] = atoms_obj
                    del json_data[reaction_key]["references"][ref_key]["atoms_json"]
    
    return json_data