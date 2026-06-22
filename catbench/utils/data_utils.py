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
import os
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


def _iter_structure_dicts(json_data):
    """Yield every leaf dict that holds a structure (has 'atoms_json' or 'ref'),
    across all CatBench data layouts (raw / star,bulk,... / references). Skips the
    top-level '_structures' map."""
    for rk, rv in json_data.items():
        if rk == "_structures" or not isinstance(rv, dict):
            continue
        if "raw" in rv and isinstance(rv["raw"], dict):
            for sv in rv["raw"].values():
                if isinstance(sv, dict):
                    yield sv
        for key in ("surface", "star", "bulk", "target"):
            if key in rv and isinstance(rv[key], dict):
                yield rv[key]
        if "references" in rv and isinstance(rv["references"], dict):
            for sv in rv["references"].values():
                if isinstance(sv, dict):
                    yield sv


def _dedup_structures_inplace(json_data):
    """Intern frame-equivalent structures into a top-level '_structures' map and
    replace each system's verbatim 'atoms_json' with a 'ref' pointer.

    Only structures that are verified ``structures_equivalent`` (and share a
    geometry+fix ``dedup_key``) collapse to one stored copy, so a resolved
    structure is physically identical (=> identical relaxed energy) to the one it
    replaced. Frame/order differences are absorbed; genuinely different structures
    (e.g. distinct relaxation minima) are kept separate.
    """
    import hashlib
    from collections import defaultdict
    from catbench.utils.structure_dedup import reuse_key, structures_equivalent

    structures = {}                 # hash -> atoms_json (canonical copy)
    reps = defaultdict(list)        # reuse_key -> [(atoms, hash)]
    for sv in _iter_structure_dicts(json_data):
        aj = sv.get("atoms_json")
        if aj is None:
            continue
        atoms = read(io.StringIO(aj), format="json")
        k = reuse_key(atoms, sv.get("energy_ref"))
        ref = None
        for rep_atoms, rep_hash in reps[k]:
            if structures_equivalent(atoms, rep_atoms):
                ref = rep_hash
                break
        if ref is None:
            # Hash the FULL reuse_key (+ bucket index) so no two distinct keys
            # can collide on a truncated prefix and silently overwrite a structure.
            ref = hashlib.md5(("%s_%d" % (k, len(reps[k]))).encode()).hexdigest()
            reps[k].append((atoms, ref))
            structures[ref] = aj
        sv["ref"] = ref
        del sv["atoms_json"]
    if structures:
        json_data["_structures"] = structures


def save_catbench_json(data_dict: Dict[str, Any], filepath: str) -> None:
    """
    Save CatBench data to JSON. ASE Atoms objects are serialized, and
    frame-equivalent structures (e.g. a clean slab or gas reference shared across
    many reactions) are interned into a top-level '_structures' map referenced by
    per-system 'ref' pointers -- smaller on disk and losslessly restored by
    ``load_catbench_json``.

    Args:
        data_dict: Dictionary containing reaction data with ASE Atoms objects
        filepath: Path to save JSON file
    """
    _write_catbench_json(data_dict, filepath, dedup=True)


def _write_catbench_json(data_dict: Dict[str, Any], filepath: str, dedup: bool = True) -> None:
    """Internal writer. ``dedup=False`` writes the verbatim ``atoms_json`` layout
    instead of interned refs -- used only by tests that assert that deduplication
    does not change anything (dedup-on == dedup-off)."""
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
    
    # Deduplicate frame-equivalent structures into a top-level "_structures" map
    # referenced by per-system "ref" pointers (1.1.1; losslessly resolved on load).
    if dedup:
        _dedup_structures_inplace(json_data)

    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Data saved to {filepath}")


def cleanup_vasp_files(directory: str, keep_files: List[str] = None, verbose: bool = True) -> None:
    """
    Clean up VASP output files, keeping only specified files.
    
    This function walks through a directory tree and removes all files except
    those specified in keep_files. Commonly used to save disk space after
    VASP calculations by keeping only essential files (CONTCAR, OSZICAR).
    
    Args:
        directory: Root directory to clean up
        keep_files: List of filenames to keep (default: ["CONTCAR", "OSZICAR"])
        verbose: Print deleted files if True
        
    Example:
        >>> cleanup_vasp_files("my_dataset/")
        >>> cleanup_vasp_files("my_dataset/", keep_files=["CONTCAR", "OSZICAR", "OUTCAR"])
    """
    if keep_files is None:
        keep_files = ["CONTCAR", "OSZICAR"]
    
    deleted_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        # Check if this directory contains VASP output files
        vasp_files = set(filenames)
        if "OSZICAR" in vasp_files or "CONTCAR" in vasp_files:
            # Delete files that are not in keep_files list
            for file in filenames:
                if file not in keep_files:
                    file_path = os.path.join(dirpath, file)
                    os.remove(file_path)
                    if verbose:
                        print(f"Deleted: {file_path}")
                    deleted_count += 1
    
    if verbose and deleted_count > 0:
        print(f"Total files deleted: {deleted_count}")
    elif verbose:
        print("No files to delete.")


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

    # Resolve deduplicated structures (1.1.1): rehydrate each "ref" pointer from
    # the top-level "_structures" map back to verbatim "atoms_json", then remove
    # the map so the reaction loop below is unchanged. Legacy files (direct
    # "atoms_json", no "_structures") are untouched -> full backward compatibility.
    structures = json_data.pop("_structures", None)
    if structures:
        for sv in _iter_structure_dicts(json_data):
            ref = sv.get("ref")
            if ref is not None:
                aj = structures.get(ref)
                if aj is None:
                    raise ValueError(
                        f"Dangling structure ref {ref!r} in {filepath} "
                        f"(missing from '_structures' map; file may be truncated/corrupt)."
                    )
                sv["atoms_json"] = aj
                del sv["ref"]

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