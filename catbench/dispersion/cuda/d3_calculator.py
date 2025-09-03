"""
Standalone D3 Calculator extracted from SevenNet
https://github.com/MDIL-SNU/SevenNet

This is a GPU-accelerated implementation of Grimme's D3 dispersion correction.
Original D3 method: J. Chem. Phys. 132, 154104 (2010)
"""

import ctypes
import os
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes


def _load(name: str) -> ctypes.CDLL:
    from torch.utils.cpp_extension import LIB_EXT, _get_build_directory, load

    # Load the library from the candidate locations

    package_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        return ctypes.CDLL(os.path.join(package_dir, f'{name}{LIB_EXT}'))
    except OSError:
        pass

    cache_dir = _get_build_directory(name, verbose=False)
    try:
        return ctypes.CDLL(os.path.join(cache_dir, f'{name}{LIB_EXT}'))
    except OSError:
        pass

    # Compile the library if it is not found

    if os.access(package_dir, os.W_OK):
        compile_dir = package_dir
    else:
        print('Warning: package directory is not writable. Using cache directory.')
        compile_dir = cache_dir

    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        print('Warning: TORCH_CUDA_ARCH_LIST is not set.')
        print('Warning: Use default CUDA architectures: 61, 70, 75, 80, 86, 89, 90')
        os.environ['TORCH_CUDA_ARCH_LIST'] = '6.1;7.0;7.5;8.0;8.6;8.9;9.0'

    load(
        name=name,
        sources=[os.path.join(package_dir, 'pair_e3gnn', 'pair_d3_for_ase.cu')],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr', '-fmad=false'],
        build_directory=compile_dir,
        verbose=True,
        is_python_module=False,
    )

    return ctypes.CDLL(os.path.join(compile_dir, f'{name}{LIB_EXT}'))


class PairD3(ctypes.Structure):
    pass  # Opaque structure; only used as a pointer


class D3Calculator(Calculator):
    """ASE calculator for accelerated D3 van der Waals (vdW) correction.

    This calculator interfaces with the `libpaird3.so` library,
    which is compiled by nvcc during the first run.
    Note: Multi-GPU parallel MD is not supported in this mode.
    """

    # Here, free_energy = energy
    implemented_properties = ['free_energy', 'energy', 'forces', 'stress']

    def __init__(
        self,
        damping_type: str = 'damp_bj',  # damp_bj, damp_zero
        functional_name: str = 'pbe',  # check the source code
        vdw_cutoff: float = 9000,  # au^2, 0.52917726 angstrom = 1 au
        cn_cutoff: float = 1600,  # au^2, 0.52917726 angstrom = 1 au
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if not torch.cuda.is_available():
            raise NotImplementedError('CPU + D3 is not implemented yet')

        self.rthr = vdw_cutoff
        self.cnthr = cn_cutoff
        self.damp_name = damping_type.lower()
        self.func_name = functional_name.lower()

        if self.damp_name not in ['damp_bj', 'damp_zero']:
            raise ValueError('Error: Invalid damping type.')

        self._init_library()

    def _idx_to_numbers(self, Z_of_atoms):
        unique_numbers = list(dict.fromkeys(Z_of_atoms))
        return unique_numbers

    def _idx_to_types(self, Z_of_atoms):
        unique_numbers = list(dict.fromkeys(Z_of_atoms))
        mapping = {num: idx + 1 for idx, num in enumerate(unique_numbers)}
        atom_types = [mapping[num] for num in Z_of_atoms]
        return atom_types

    def _convert_domain_ase2lammps(self, cell):
        qtrans, ltrans = np.linalg.qr(cell.T, mode='complete')
        lammps_cell = ltrans.T
        signs = np.sign(np.diag(lammps_cell))
        lammps_cell = lammps_cell * signs
        qtrans = qtrans * signs
        lammps_cell = lammps_cell[(0, 1, 2, 1, 2, 2), (0, 1, 2, 0, 0, 1)]
        rotator = qtrans.T
        return lammps_cell, rotator

    def _stress2tensor(self, stress):
        tensor = np.array(
            [
                [stress[0], stress[3], stress[4]],
                [stress[3], stress[1], stress[5]],
                [stress[4], stress[5], stress[2]],
            ]
        )
        return tensor

    def _tensor2stress(self, tensor):
        stress = -np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                tensor[1, 2],
                tensor[0, 2],
                tensor[0, 1],
            ]
        )
        return stress

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            raise ValueError('No atoms to evaluate')

        if atoms.get_cell().sum() == 0:
            print(
                'Warning: D3Calculator requires a cell.\n'
                'Warning: An orthogonal cell large enough is generated.'
            )
            positions = atoms.get_positions()
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            max_cutoff = np.sqrt(max(self.rthr, self.cnthr)) * 0.52917726

            cell_lengths = max_pos - min_pos + max_cutoff + 1.0  # extra margin
            cell = np.eye(3) * cell_lengths

            atoms.set_cell(cell)
            atoms.set_pbc([True, True, True])  # for minus positions

        cell, rotator = self._convert_domain_ase2lammps(atoms.get_cell())

        Z_of_atoms = atoms.get_atomic_numbers()
        natoms = len(atoms)
        ntypes = len(set(Z_of_atoms))
        types = (ctypes.c_int * natoms)(*self._idx_to_types(Z_of_atoms))

        positions = atoms.get_positions() @ rotator.T
        x_flat = (ctypes.c_double * (natoms * 3))(*positions.flatten())

        atomic_numbers = (ctypes.c_int * ntypes)(*self._idx_to_numbers(Z_of_atoms))

        boxlo = (ctypes.c_double * 3)(0.0, 0.0, 0.0)
        boxhi = (ctypes.c_double * 3)(cell[0], cell[1], cell[2])
        xy = cell[3]
        xz = cell[4]
        yz = cell[5]
        xperiodic, yperiodic, zperiodic = atoms.get_pbc()

        lib = self._lib
        assert lib is not None
        lib.pair_set_atom(self.pair, natoms, ntypes, types, x_flat)

        xperiodic = xperiodic.astype(int)
        yperiodic = yperiodic.astype(int)
        zperiodic = zperiodic.astype(int)
        lib.pair_set_domain(
            self.pair, xperiodic, yperiodic, zperiodic, boxlo, boxhi, xy, xz, yz
        )

        lib.pair_run_settings(
            self.pair,
            self.rthr,
            self.cnthr,
            self.damp_name.encode('utf-8'),
            self.func_name.encode('utf-8'),
        )

        lib.pair_run_coeff(self.pair, atomic_numbers)
        lib.pair_run_compute(self.pair)

        result_E = lib.pair_get_energy(self.pair)

        result_F_ptr = lib.pair_get_force(self.pair)
        result_F_size = natoms * 3
        result_F = np.ctypeslib.as_array(
            result_F_ptr, shape=(result_F_size,)
        ).reshape((natoms, 3))
        result_F = np.array(result_F)
        result_F = result_F @ rotator

        result_S = lib.pair_get_stress(self.pair)
        result_S = np.array(result_S.contents)
        result_S = (
            self._tensor2stress(rotator.T @ self._stress2tensor(result_S) @ rotator)
            / atoms.get_volume()
        )

        self.results = {
            'free_energy': result_E,
            'energy': result_E,
            'forces': result_F,
            'stress': result_S,
        }

    def __getstate__(self):
        """Exclude ctypes objects during pickle"""
        state = self.__dict__.copy()
        # Remove ctypes objects
        state.pop('_lib', None)
        state.pop('pair', None)
        return state
    
    def __setstate__(self, state):
        """Reinitialize after unpickle"""
        self.__dict__.update(state)
        # Reload library
        self._init_library()
    
    def _init_library(self):
        """Initialize library (reusable code)"""
        self._lib = _load('pair_d3')
        
        self._lib.pair_init.restype = ctypes.POINTER(PairD3)
        self.pair = self._lib.pair_init()
        
        self._lib.pair_set_atom.argtypes = [
            ctypes.POINTER(PairD3),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.pair_set_atom.restype = None
        
        self._lib.pair_set_domain.argtypes = [
            ctypes.POINTER(PairD3),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self._lib.pair_set_domain.restype = None
        
        self._lib.pair_run_settings.argtypes = [
            ctypes.POINTER(PairD3),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self._lib.pair_run_settings.restype = None
        
        self._lib.pair_run_coeff.argtypes = [
            ctypes.POINTER(PairD3),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.pair_run_coeff.restype = None
        
        self._lib.pair_run_compute.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_run_compute.restype = None
        
        self._lib.pair_get_energy.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_get_energy.restype = ctypes.c_double
        
        self._lib.pair_get_force.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_get_force.restype = ctypes.POINTER(ctypes.c_double)
        
        self._lib.pair_get_stress.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_get_stress.restype = ctypes.POINTER(ctypes.c_double * 6)
        
        self._lib.pair_fin.argtypes = [ctypes.POINTER(PairD3)]
        self._lib.pair_fin.restype = None
    
    def __del__(self):
        if hasattr(self, '_lib') and self._lib is not None:
            self._lib.pair_fin(self.pair)
            self._lib = None
            self.pair = None