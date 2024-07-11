from dpdata.unit import econvs
from typing import List

from ase.geometry.cell import cell_to_cellpar
from ase import Atoms
import numpy as np

import ase.io
import dpdata
import re


from .util import LammpsData
from ai2_kit.core.log import get_logger

logger = get_logger(__name__)


def set_dplr_ext_from_cp2k_output(dp_sys: dpdata.LabeledSystem,
                                  cp2k_output: str,
                                  wannier_file: str,
                                  type_map: List[str],
                                  sys_charge_map: List[int],
                                  model_charge_map: List[int],
                                  ewald_h: float,
                                  ewald_beta: float,
                                  ext_efield,
                                  sel_type: List[int],
                                  ):

    wannier_atoms = ase.io.read(wannier_file)
    with open(cp2k_output, 'r') as fp:
        raw_energy = get_cp2k_output_total_energy(fp)

    ext_efield = np.reshape(ext_efield, [1, 3])
    natoms = dp_sys.get_natoms()
    nframes = dp_sys.get_nframes()
    assert nframes == 1, 'Only support one frame'

    symbols = np.array(dp_sys.data["atom_names"])[dp_sys.data["atom_types"]]
    sel_ids = get_sel_ids(dp_sys, type_map, sel_type)

    atomic_dipole, extended_coords = get_atomic_dipole(dp_sys, sel_ids, wannier_atoms)
    # get extended charges
    extended_charges = np.zeros(len(extended_coords))
    for symbol, ion_charge in zip(type_map, sys_charge_map):
        extended_charges[np.where(symbols == symbol)[0]] = ion_charge
    sel_symbols = symbols[sel_ids]
    for ii, wannier_charge in zip(sel_type, model_charge_map):
        sel_symbol = type_map[ii]
        extended_charges[np.where(sel_symbols == sel_symbol)[0] + natoms] = wannier_charge
    total_dipole = np.sum(extended_coords * extended_charges.reshape(-1, 1), axis=0)
    energy = raw_energy - np.inner(total_dipole, ext_efield.reshape(3))
    dp_sys.data['energies'] = np.reshape(energy, [-1])
    pbc_efield = get_pbc_atomic_efield(dp_sys, extended_coords, extended_charges, ewald_h, ewald_beta)
    # natoms * 3
    _efield = pbc_efield + ext_efield
    aparam = np.linalg.norm(_efield, axis=-1)
    efield = np.array(_efield) / aparam.reshape(natoms, 1)

    dp_sys.data['ext_efield'] = ext_efield.reshape(nframes, 3)
    dp_sys.data['efield'] = efield.reshape(nframes, natoms, 3)
    dp_sys.data['aparam'] = aparam.reshape(nframes, natoms, -1)
    dp_sys.data['atomic_dipole'] = atomic_dipole.reshape(nframes, -1)
    return dp_sys

def get_cp2k_output_total_energy(fp):
    start_pattern = r"  Total charge density g-space grids:"
    end_pattern = r"  Total energy:"
    start_pattern = re.compile(start_pattern)
    end_pattern = re.compile(end_pattern)

    flag = False
    data_lines = []
    nloop = 0
    for line in fp:
        line = line.strip('\n')
        if start_pattern.match(line):
            flag = True
        elif flag and end_pattern.match(line):
            flag = False
            nloop += 1

        if flag is True:
            data_lines.append(line)
    data_lines = np.reshape(data_lines, (nloop, -1))

    tot_e = 0.
    # grep the data from the last iteration
    for kw in data_lines[-1, 2:-1]:
        kw = kw.split(":")
        k = kw[0].strip(' ')
        v = float(kw[1]) * econvs["hartree"]
        if k != "Fermi energy":
            tot_e += v
    return tot_e


def get_sel_ids(dp_sys, type_map, sel_type):
    symbols = np.array(dp_sys.data["atom_names"])[dp_sys.data["atom_types"]]
    sel_ids = [np.where(symbols == type_map[ii])[0] for ii in sel_type]
    return np.concatenate(sel_ids)


def get_atomic_dipole(dp_sys, sel_ids, wannier_atoms, wannier_cutoff = 1.0):
    from MDAnalysis.lib.distances import distance_array, minimize_vectors

    coords = dp_sys.data['coords'].reshape(-1, 3)
    cellpar = cell_to_cellpar(dp_sys.data['cells'].reshape(3, 3))
    extended_coords = coords.copy()

    ref_coords = coords[sel_ids].reshape(-1, 3)
    e_coords = wannier_atoms.get_positions()
    dist_mat = distance_array(ref_coords, e_coords, box=cellpar)
    atomic_dipole = []
    for ii, dist_vec in enumerate(dist_mat):
        mask = (dist_vec < wannier_cutoff)
        cn = np.sum(mask)
        if cn != 4:
            raise ValueError(f'wannier atoms {ii} has {cn} atoms in the cutoff range')
        wc_coord_rel = e_coords[mask] - ref_coords[ii]
        wc_coord_rel = minimize_vectors(wc_coord_rel, box=cellpar)
        _atomic_dipole = wc_coord_rel.mean(axis=0)
        atomic_dipole.append(_atomic_dipole)
        wc_coord = _atomic_dipole + ref_coords[ii]
        extended_coords = np.concatenate((extended_coords, wc_coord.reshape(1, 3)), axis=0)
    atomic_dipole = np.reshape(atomic_dipole, (-1, 3))
    assert atomic_dipole.shape[0] == len(sel_ids)
    return atomic_dipole, extended_coords


def get_pbc_atomic_efield(dp_sys,
                          extended_coords,
                          extended_charges,
                          ewald_h,
                          ewald_beta,
                          ):
    from deepmd.infer.ewald_recp import EwaldRecp

    er = EwaldRecp(ewald_h, ewald_beta)
    natoms = dp_sys.get_natoms()
    cell = dp_sys.data['cells']
    e, f, v = er.eval(extended_coords.reshape(1, -1),
                        extended_charges.reshape(1, -1),
                        cell.reshape(1, -1))
    extended_efield = f.reshape(-1, 3) / extended_charges.reshape(-1, 1)
    # natoms * 3
    pbc_efield = extended_efield[:natoms]
    return pbc_efield


def get_sel_type(model_path) -> List[int]:
    try:
        from deepmd.infer import DeepDipole
    except ImportError:
        raise ImportError('deepmd-kit is not installed')

    dp = DeepDipole(model_path)
    return [t for t in dp.tselt]


def build_sel_type_assertion(sel_type, model_path: str, py_cmd='python'):
    return f'''{py_cmd} -c "from deepmd.infer import DeepDipole;dp = DeepDipole({repr(model_path)});assert{repr(sel_type)}==[t for t in dp.tselt]"'''


def dump_dplr_lammps_data(fp, atoms: Atoms, type_map: List[str], sel_type: List[int],
                          sys_charge_map: List[float], model_charge_map: List[float]):

    """
    dump atoms to LAMMPS data file for DPLR
    the naming convention of params follows Deepmd-Kit's

    about dplr: https://docs.deepmodeling.com/projects/deepmd/en/master/model/dplr.html
    about fitting tensor: https://docs.deepmodeling.com/projects/deepmd/en/master/model/train-fitting-tensor.html

    :param fp: file pointer
    :param type_map: the type map of atom type, for example, [O,H]
    :param sel_type: the selected type of atom, for example, [0] means atom type 0, aka O is selected
    :param sys_charge_map: the charge map of atom in system, for example, [6, 1]
    :param model_charge_map: the charge map of atom in model, for example, [-8]
    """

    if len(type_map) != len(sys_charge_map):
        raise ValueError(f'type_map {type_map} and sys_charge_map {sys_charge_map} must have the same length')

    if len(sel_type) != len(model_charge_map):
        raise ValueError(f'sel_type {sel_type} and model_charge_map {model_charge_map} must have the same length')

    new_atoms = atoms.copy()
    vtypes = get_unused_symbols(type_map, len(sel_type))

    r_atom_ids = []
    v_atom_ids = []

    # add virtual atoms
    for atype, vtype in zip(sel_type, vtypes):
        # create virtual atoms from the original atom type
        v_atoms = atoms[atoms.symbols == type_map[atype]]
        v_atoms.set_chemical_symbols([vtype] * len(v_atoms))  # type: ignore

        # create bonds between atom and its virtual atoms
        r_atom_ids.extend(np.where(atoms.symbols == type_map[atype])[0])
        v_atom_ids.extend(np.arange(len(v_atoms)) + len(new_atoms))  # type: ignore
        # this should be last
        new_atoms += v_atoms

    # build charges
    charges = np.zeros(len(new_atoms))
    for atype, charge in zip(type_map + vtypes, sys_charge_map + model_charge_map):
        charges[new_atoms.symbols==atype] = charge
    # build bonds
    n_bonds = len(r_atom_ids)
    # note: the index of lammps start from 1
    bonds = np.array([
        np.arange(n_bonds) + 1,  # bond id
        [1] * n_bonds,  # bond type
        np.array(r_atom_ids) + 1,  # bond left
        np.array(v_atom_ids) + 1,  # bond right
    ], dtype=int).T

    # write lammps data
    lmp_data = LammpsData(new_atoms)
    lmp_data.charges = charges
    lmp_data.bonds = bonds
    lmp_data.write(fp, atom_style='full', specorder=type_map + vtypes)

def get_unused_symbols(used_symbols: List[str], size: int):
    """
    Get unused symbols from ase

    :param used_symbols: symbols that are already used
    :param size: number of unused symbols to get
    """
    from ase.data import chemical_symbols

    unused_symbols = []
    for symbol in reversed(chemical_symbols):
        if symbol not in used_symbols:
            unused_symbols.append(symbol)
        if len(unused_symbols) == size:
            break
    return unused_symbols
