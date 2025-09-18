"""
Utility functions for MLIP calculations and benchmarking.

This module contains helper functions for energy calculations, structure manipulation,
and data processing used by the main CatbenchCalculation class.
"""

import json
import os
import time
from copy import deepcopy

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import LBFGS, BFGS, GPMin, FIRE, MDMin, BFGSLineSearch, LBFGSLineSearch


def convert_trajectory(filename):
    """Convert trajectory file to extxyz format."""
    images = read(filename, index=":")
    os.remove(filename)
    write(filename, images, format="extxyz")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def energy_cal_gas(
    calculator,
    atoms_origin,
    f_crit_relax,
    save_path,
    optimizer,
    log_path=None,
    filename=None,
):
    """Calculate energy for gas-phase molecules."""
    optimizer_classes = {
        "LBFGS": LBFGS,
        "BFGS": BFGS,
        "GPMin": GPMin,
        "FIRE": FIRE,
        "MDMin": MDMin,
        "BFGSLineSearch": BFGSLineSearch,
        "LBFGSLineSearch": LBFGSLineSearch,
    }

    if optimizer in optimizer_classes:
        opt_class = optimizer_classes[optimizer]
        atoms = deepcopy(atoms_origin)
        atoms.calc = calculator
        atomic_numbers = atoms.get_atomic_numbers()
        max_atomic_number = np.max(atomic_numbers)
        max_atomic_number_indices = [
            i for i, num in enumerate(atomic_numbers) if num == max_atomic_number
        ]
        fixed_atom_index = np.random.choice(max_atomic_number_indices)
        c = FixAtoms(indices=[fixed_atom_index])
        atoms.set_constraint(c)
        tags = np.ones(len(atoms))
        atoms.set_tags(tags)

        if save_path is not None:
            write(save_path, atoms)

        if log_path is not None and filename is not None:
            logfile = open(log_path, "w", buffering=1)
            logfile.write("######################\n")
            logfile.write("##  MLIP relax starts  ##\n")
            logfile.write("######################\n")
            logfile.write("\nStep 1. Relaxing\n")

            opt = opt_class(atoms, logfile=logfile, trajectory=filename)
            time_init = time.time()
            opt.run(fmax=f_crit_relax, steps=500)
            elapsed_time = time.time() - time_init

            convert_trajectory(filename)
            logfile.write("Done!\n")
            logfile.write(f"\nElapsed time: {elapsed_time} s\n\n")
            logfile.write("###############################\n")
            logfile.write("##  Relax terminated normally  ##\n")
            logfile.write("###############################\n")
            logfile.close()
        else:
            # Run without saving log/trajectory
            opt = opt_class(atoms, logfile=None, trajectory=None)
            time_init = time.time()
            opt.run(fmax=f_crit_relax, steps=500)
            elapsed_time = time.time() - time_init

        return atoms, atoms.get_potential_energy()


def energy_cal_single(calculator, atoms_origin):
    """Calculate single-point energy without optimization."""
    atoms = deepcopy(atoms_origin)
    atoms.calc = calculator
    tags = np.ones(len(atoms))
    atoms.set_tags(tags)
    return atoms.get_potential_energy()


def energy_cal(
    calculator,
    atoms_origin,
    f_crit_relax,
    n_crit_relax,
    damping,
    z_target,
    optimizer,
    logfile=None,
    filename=None,
):
    """Calculate energy with structure optimization."""
    atoms = deepcopy(atoms_origin)
    atoms.calc = calculator
    tags = np.ones(len(atoms))
    atoms.set_tags(tags)
    if z_target != None:
        atoms.set_constraint(fixatom(atoms, z_target))
    # Record initial energy before optimization
    initial_energy = atoms.get_potential_energy()

    optimizer_classes = {
        "LBFGS": LBFGS,
        "BFGS": BFGS,
        "GPMin": GPMin,
        "FIRE": FIRE,
        "MDMin": MDMin,
        "BFGSLineSearch": BFGSLineSearch,
        "LBFGSLineSearch": LBFGSLineSearch,
    }

    if optimizer in optimizer_classes:
        opt_class = optimizer_classes[optimizer]

        if logfile is None or filename is None:
            # Run without saving log/trajectory
            opt = opt_class(atoms, logfile=None, trajectory=None)
            time_init = time.time()
            opt.run(fmax=f_crit_relax, steps=n_crit_relax)
            elapsed_time = time.time() - time_init
        else:
            logfile = open(logfile, "w", buffering=1)
            logfile.write("######################\n")
            logfile.write("##  MLIP relax starts  ##\n")
            logfile.write("######################\n")
            logfile.write("\nStep 1. Relaxing\n")
            opt = opt_class(atoms, logfile=logfile, trajectory=filename)
            time_init = time.time()
            opt.run(fmax=f_crit_relax, steps=n_crit_relax)
            elapsed_time = time.time() - time_init
            convert_trajectory(filename)
            logfile.write("Done!\n")
            logfile.write(f"\nElapsed time: {elapsed_time} s\n\n")
            logfile.write("###############################\n")
            logfile.write("##  Relax terminated normally  ##\n")
            logfile.write("###############################\n")
            logfile.close()
        final_energy = atoms.get_potential_energy()
        energy_change = final_energy - initial_energy  # Negative = stabilization

    return final_energy, opt.nsteps, atoms, elapsed_time, energy_change


def fixatom(atoms, z_target):
    """Fix atoms below specified z-coordinate."""
    indices_to_fix = [atom.index for atom in atoms if atom.position[2] < z_target]
    const = FixAtoms(indices=indices_to_fix)
    return const


def calc_displacement(atoms1, atoms2, z_target=None):
    """
    Calculate displacement statistics between two structures.
    
    Args:
        atoms1: Initial atomic structure
        atoms2: Final atomic structure  
        z_target: Z-coordinate threshold for atom fixing (if None, all atoms are considered mobile)
        
    Returns:
        dict: Dictionary containing displacement statistics:
            - max_disp: Maximum displacement of any atom
            - mae_mobile: Mean Absolute Error of mobile (free) atoms displacement
            - rmsd_mobile: Root Mean Square Deviation of mobile (free) atoms displacement
    """
    positions1 = atoms1.get_positions()
    positions2 = atoms2.get_positions()
    
    # Calculate displacement for each atom
    displacement_magnitudes = np.linalg.norm(positions2 - positions1, axis=1)
    
    # Maximum displacement of any atom
    max_disp = np.max(displacement_magnitudes)
    
    # If z_target is provided, calculate MAE and RMSD for mobile (free) atoms only
    if z_target is not None:
        # Identify mobile (non-fixed) atoms - use original positions for z comparison
        mobile_mask = atoms1.positions[:, 2] >= z_target
        
        if np.sum(mobile_mask) > 0:
            mobile_displacements = displacement_magnitudes[mobile_mask]
            mae_mobile = np.mean(mobile_displacements)  # Mean Absolute Error
            rmsd_mobile = np.sqrt(np.mean(mobile_displacements**2))  # Root Mean Square Deviation
        else:
            mae_mobile = 0.0
            rmsd_mobile = 0.0
    else:
        # If no z_target, use all atoms
        mae_mobile = np.mean(displacement_magnitudes)
        rmsd_mobile = np.sqrt(np.mean(displacement_magnitudes**2))
    
    return {
        "max_disp": max_disp,
        "mae_mobile": mae_mobile,
        "rmsd_mobile": rmsd_mobile
    }


def find_median_index(arr):
    """Find index of median value in array."""
    orig_arr = deepcopy(arr)
    sorted_arr = sorted(arr)
    length = len(sorted_arr)
    median_index = (length - 1) // 2
    median_value = sorted_arr[median_index]
    for i, num in enumerate(orig_arr):
        if num == median_value:
            return i, median_value


def fix_z(atoms, rate_fix):
    """Calculate z-coordinate for fixing atoms based on rate."""
    if rate_fix != None:
        z_max = max(atoms.positions[:, 2])
        z_min = min(atoms.positions[:, 2])
        z_target = z_min + rate_fix * (z_max - z_min)
        return z_target
    else:
        return None