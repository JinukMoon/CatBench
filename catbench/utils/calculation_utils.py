"""
Utility functions for MLIP calculations and benchmarking.

This module contains helper functions for energy calculations, structure manipulation,
and data processing used by the main CatbenchCalculation class.
"""

import io
import json
import os
import time
from copy import deepcopy

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.geometry import find_mic
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
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
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

    if optimizer not in optimizer_classes:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. Valid: {list(optimizer_classes)}"
        )

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
            # Same I/O decoupling as energy_cal: collect trajectory frames and the
            # log in memory during the run, flush to disk after the timer (keeps
            # per-step NFS writes out of opt.run()).
            log_buf = io.StringIO()
            log_buf.write("######################\n")
            log_buf.write("##  MLIP relax starts  ##\n")
            log_buf.write("######################\n")
            log_buf.write("\nStep 1. Relaxing\n")
            frames = []

            def _snapshot(a=atoms, fr=frames):
                snap = a.copy()
                snap.calc = SinglePointCalculator(
                    snap, energy=a.get_potential_energy(), forces=a.get_forces()
                )
                fr.append(snap)

            opt = opt_class(atoms, logfile=log_buf, trajectory=None)
            opt.attach(_snapshot, interval=1)
            time_init = time.time()
            opt.run(fmax=f_crit_relax, steps=500)
            elapsed_time = time.time() - time_init

            # --- excluded from elapsed_time ---
            write(filename, frames, format="extxyz")
            log_buf.write("Done!\n")
            log_buf.write(f"\nElapsed time: {elapsed_time} s\n\n")
            log_buf.write("###############################\n")
            log_buf.write("##  Relax terminated normally  ##\n")
            log_buf.write("###############################\n")
            with open(log_path, "w") as _lf:
                _lf.write(log_buf.getvalue())
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
    fixed_indices,
    optimizer,
    logfile=None,
    filename=None,
):
    """Calculate energy with structure optimization."""
    atoms = deepcopy(atoms_origin)
    atoms.calc = calculator
    tags = np.ones(len(atoms))
    atoms.set_tags(tags)
    if fixed_indices is not None:
        atoms.set_constraint(FixAtoms(indices=list(fixed_indices)))
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

    if optimizer not in optimizer_classes:
        raise ValueError(
            f"Unknown optimizer: {optimizer}. Valid: {list(optimizer_classes)}"
        )

    if optimizer in optimizer_classes:
        opt_class = optimizer_classes[optimizer]

        if logfile is None or filename is None:
            # Run without saving log/trajectory
            opt = opt_class(atoms, logfile=None, trajectory=None)
            time_init = time.time()
            opt.run(fmax=f_crit_relax, steps=n_crit_relax)
            elapsed_time = time.time() - time_init
        else:
            # Timing integrity: keep ALL disk I/O (trajectory + log) OUTSIDE the
            # timer. The ASE optimizer, given trajectory=filename / a file logfile,
            # writes to disk every step inside opt.run() -- on a network filesystem
            # those per-step writes contaminate elapsed_time (and thus
            # time_per_step) with filesystem speed instead of model compute. Here
            # the trajectory frames and the log are collected in memory during the
            # run (microseconds) and flushed to disk only after the timer stops.
            log_buf = io.StringIO()
            log_buf.write("######################\n")
            log_buf.write("##  MLIP relax starts  ##\n")
            log_buf.write("######################\n")
            log_buf.write("\nStep 1. Relaxing\n")
            frames = []

            def _snapshot(a=atoms, fr=frames):
                # copy() drops the calculator, so re-attach the (already cached)
                # energy/forces as a SinglePointCalculator to keep the saved
                # trajectory faithful; these reads are cached -> microseconds.
                snap = a.copy()
                snap.calc = SinglePointCalculator(
                    snap, energy=a.get_potential_energy(), forces=a.get_forces()
                )
                fr.append(snap)

            opt = opt_class(atoms, logfile=log_buf, trajectory=None)
            opt.attach(_snapshot, interval=1)
            time_init = time.time()
            opt.run(fmax=f_crit_relax, steps=n_crit_relax)
            elapsed_time = time.time() - time_init

            # --- everything below is excluded from elapsed_time ---
            write(filename, frames, format="extxyz")
            log_buf.write("Done!\n")
            log_buf.write(f"\nElapsed time: {elapsed_time} s\n\n")
            log_buf.write("###############################\n")
            log_buf.write("##  Relax terminated normally  ##\n")
            log_buf.write("###############################\n")
            with open(logfile, "w") as _lf:
                _lf.write(log_buf.getvalue())
        final_energy = atoms.get_potential_energy()
        energy_change = final_energy - initial_energy  # Negative = stabilization

    return final_energy, opt.nsteps, atoms, elapsed_time, energy_change


def get_fixed_indices(atoms):
    """
    Determine the indices of fixed atoms for a structure from its stored
    FixAtoms constraints (data-defined fixing only).

    Args:
        atoms: ASE Atoms object

    Returns:
        list[int]: Sorted, de-duplicated indices of fixed atoms (may be empty)
    """
    indices = []
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            indices.extend(int(i) for i in c.get_indices())
    return sorted(set(indices))


def calc_displacement(atoms1, atoms2, fixed_indices=None):
    """
    Calculate displacement statistics between two structures.

    Args:
        atoms1: Initial atomic structure
        atoms2: Final atomic structure
        fixed_indices: Indices of fixed atoms. Mobile atoms are the complement of this set.
                       If None, all atoms are considered mobile.

    Returns:
        dict: Dictionary containing displacement statistics:
            - max_disp: Maximum displacement of any atom
            - mae_mobile: Mean Absolute Error of mobile (free) atoms displacement
            - rmsd_mobile: Root Mean Square Deviation of mobile (free) atoms displacement
    """
    positions1 = atoms1.get_positions()
    positions2 = atoms2.get_positions()

    # Calculate displacement for each atom.
    # Apply the minimum-image convention so displacements are correct for
    # non-orthogonal cells and atoms that wrapped across a periodic boundary.
    diffs = positions2 - positions1
    mic_diffs, _ = find_mic(diffs, atoms1.cell, atoms1.pbc)
    displacement_magnitudes = np.linalg.norm(mic_diffs, axis=1)

    # Maximum displacement of any atom
    max_disp = np.max(displacement_magnitudes)

    # If fixed_indices is provided, calculate MAE and RMSD for mobile (free) atoms only
    if fixed_indices is not None:
        # Mobile (non-fixed) atoms are the complement of fixed_indices
        mobile_mask = np.ones(len(atoms1), dtype=bool)
        if len(fixed_indices) > 0:
            mobile_mask[list(fixed_indices)] = False

        if np.sum(mobile_mask) > 0:
            mobile_displacements = displacement_magnitudes[mobile_mask]
            mae_mobile = np.mean(mobile_displacements)  # Mean Absolute Error
            rmsd_mobile = np.sqrt(np.mean(mobile_displacements**2))  # Root Mean Square Deviation
        else:
            mae_mobile = 0.0
            rmsd_mobile = 0.0
    else:
        # If no fixed_indices, use all atoms
        mae_mobile = np.mean(displacement_magnitudes)
        rmsd_mobile = np.sqrt(np.mean(displacement_magnitudes**2))
    
    return {
        "max_disp": max_disp,
        "mae_mobile": mae_mobile,
        "rmsd_mobile": rmsd_mobile
    }


def find_median_index(arr):
    """Find the original index of the median-rank value in an array.

    NaN-safe and deterministic: NaN values sort to the end (numpy argsort
    behavior), so a diverged energy never yields a None return. The element at
    the median RANK is returned (not the first ``== median_value`` match), so
    under duplicate energies ties now break by rank position, which may give a
    slightly different median_num than the previous implementation. This is the
    correct, deterministic behavior.

    Returns:
        tuple[int, float]: (original index of the median-rank element, its value)
    """
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        raise ValueError("find_median_index: empty array")
    order = np.argsort(a, kind="stable")   # NaN sorts to the end deterministically
    median_index = int(order[(a.size - 1) // 2])
    return median_index, float(a[median_index])