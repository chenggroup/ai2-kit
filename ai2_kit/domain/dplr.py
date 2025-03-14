from typing import List, Optional

from ase.geometry.cell import cell_to_cellpar
from ase import Atoms
import numpy as np

import ase.io
import dpdata
import os
import glob

from .util import LammpsData
from ai2_kit.core.log import get_logger

logger = get_logger(__name__)


def dpdata_read_cp2k_dplr_data(
    cp2k_dir: str,
    cp2k_output: str,
    wannier_file: str,
    type_map: List[str],
    sel_type: List[int],
    wannier_cutoff: float = 1.0,
    wannier_spread_file: Optional[str] = None,
    model_charge_map: Optional[List[int]] = None,
    export_atomic_weight: bool = False,
):
    """
    Gnereate dpdata from cp2k output and wannier file for DPLR

    :param cp2k_dir: the directory of cp2k output and wannier file
    :param cp2k_output: the cp2k output file
    :param wannier_file: the wannier file
    :param type_map: the type map of atom type, for example, [O,H]
    :param sel_type: the selected type of atom, for example, [0] means atom type 0, aka O is selected
    :param wannier_cutoff: the cutoff to allocate wannier centers around atoms
    :param wannier_spread_file: the wannier spread file, if provided, the spread data will be added to dp_sys
    :param model_charge_map: the charge map of wannier in model, for example, [-8]
    :param export_atomic_weight: whether to export atomic weight rather than return None when getting exception

    :return dp_sys: dpdata.LabeledSystem
        In addition to the common energy data, atomic_dipole data is added.
    """
    cp2k_output = os.path.join(cp2k_dir, cp2k_output)
    wannier_file = os.path.join(cp2k_dir, wannier_file)
    if wannier_spread_file is not None:
        assert (
            model_charge_map is None
        ), "model_charge_map is not supported when collecting wannier_spread"
        assert (
            not export_atomic_weight
        ), "export_atomic_weight is not supported when collecting wannier_spread"

    wannier_spread_file = (
        os.path.join(cp2k_dir, wannier_spread_file) if wannier_spread_file else None
    )
    dp_sys = dpdata.LabeledSystem(cp2k_output, fmt="cp2k/output")
    try:
        dp_sys = set_dplr_ext_from_cp2k_output(
            dp_sys,
            wannier_file,
            type_map,
            sel_type,
            wannier_cutoff,
            wannier_spread_file,
            model_charge_map,
            export_atomic_weight,
        )
    except:
        dp_sys = None

    return dp_sys


def set_dplr_ext_from_cp2k_output(
    dp_sys: dpdata.LabeledSystem,
    wannier_file: str,
    type_map: List[str],
    sel_type: List[int],
    wannier_cutoff: float = 1.0,
    wannier_spread_file: Optional[str] = None,
    model_charge_map: Optional[List[int]] = None,
    export_atomic_weight: bool = False,
):
    wannier_atoms = ase.io.read(wannier_file)
    # assert np.all(wannier_atoms.symbols == "X"), (
    #     "%s should include Wannier centres only" % wannier_file
    # )
    mask = wannier_atoms.symbols == "X"  # type: ignore
    wannier_atoms = wannier_atoms[mask]

    natoms = dp_sys.get_natoms()
    nframes = dp_sys.get_nframes()
    if nframes == 0:
        return None
    assert nframes == 1, "Only support one frame"

    if export_atomic_weight:
        dp_sys.data["atomic_weight"] = np.ones([nframes, natoms, 1])

    # symbols = np.array(dp_sys.data["atom_names"])[dp_sys.data["atom_types"]]
    sel_ids = get_sel_ids(dp_sys, type_map, sel_type)
    cns_ref = get_ref_cns(dp_sys, type_map, sel_type, model_charge_map)
    atomic_dipole, extended_coords, wannier_spread = get_atomic_dipole(
        dp_sys,
        sel_ids,
        wannier_atoms,
        wannier_cutoff,
        wannier_spread_file,
        cns_ref,
    )
    atomic_dipole_reformat = np.zeros((nframes, natoms, 3))
    atomic_dipole_reformat[:, sel_ids, :] = atomic_dipole.reshape([nframes, -1, 3])
    atomic_dipole = atomic_dipole_reformat
    dp_sys.data["atomic_dipole"] = atomic_dipole.reshape([nframes, natoms, 3])
    try:
        wannier_spread_reformat = np.zeros((nframes, natoms, 4))
        if len(wannier_spread) > 0:
            wannier_spread_reformat[:, sel_ids] = wannier_spread.reshape(
                [nframes, -1, 4]
            )
            wannier_spread = wannier_spread_reformat
            dp_sys.data["wannier_spread"] = np.reshape(
                wannier_spread, [nframes, natoms, 4]
            )
    except:
        pass
    return dp_sys


def get_sel_ids(dp_sys, type_map, sel_type):
    symbols = np.array(dp_sys.data["atom_names"])[dp_sys.data["atom_types"]]
    # sel_ids = [np.where(symbols == type_map[ii])[0] for ii in sel_type]
    sel_ids = np.where(np.isin(symbols, np.array(type_map)[sel_type]))[0]
    return sel_ids


def get_ref_cns(dp_sys, type_map, sel_type, model_charge_map):
    if model_charge_map is None:
        return None
    else:
        cns_ref = []
        symbols = np.array(dp_sys.data["atom_names"])[dp_sys.data["atom_types"]]
        for s in symbols:
            for ii, idx in enumerate(sel_type):
                if s == type_map[idx]:
                    cns_ref.append(model_charge_map[ii])
        cns_ref = np.array(cns_ref) / (-2)
        return np.array(cns_ref, dtype=int)


def get_atomic_dipole(
    dp_sys,
    sel_ids,
    wannier_atoms,
    wannier_cutoff=1.0,
    wannier_spread_file=None,
    cns_ref=None,
):
    from MDAnalysis.lib.distances import distance_array, minimize_vectors

    coords = dp_sys.data["coords"].reshape(-1, 3)
    cellpar = cell_to_cellpar(dp_sys.data["cells"].reshape(3, 3))
    extended_coords = coords.copy()

    if wannier_spread_file:
        full_wannier_spread = read_wannier_spread(wannier_spread_file)[:, -1]
    else:
        full_wannier_spread = None

    export_atomic_weight = "atomic_weight" in dp_sys.data

    ref_coords = coords[sel_ids].reshape(-1, 3)
    e_coords = wannier_atoms.get_positions()
    dist_mat = distance_array(ref_coords, e_coords, box=cellpar)
    atomic_dipole = []
    wannier_spread = []
    mlwc_ids = []
    if cns_ref is None:
        cns_ref = [4] * len(sel_ids)
    for ii, dist_vec in enumerate(dist_mat):
        mask = dist_vec < wannier_cutoff
        cn = np.sum(mask)
        if cn != cns_ref[ii]:
            if not export_atomic_weight:
                raise ValueError(
                    f"wannier atoms {ii} has {cn} atoms in the cutoff range"
                )
            dp_sys.data["atomic_weight"][0, sel_ids[ii], 0] = 0
        mlwc_ids.append(np.where(mask)[0])
        wc_coord_rel = e_coords[mask] - ref_coords[ii]
        if full_wannier_spread is not None:
            wannier_spread.append(full_wannier_spread[mask])
        wc_coord_rel = minimize_vectors(wc_coord_rel, box=cellpar)
        _atomic_dipole = wc_coord_rel.mean(axis=0)
        atomic_dipole.append(_atomic_dipole)
        wc_coord = _atomic_dipole + ref_coords[ii]
        extended_coords = np.concatenate(
            (extended_coords, wc_coord.reshape(1, 3)), axis=0
        )
    mlwc_ids = np.concatenate(mlwc_ids)
    # exclude double counting
    assert len(np.unique(mlwc_ids)) == len(mlwc_ids)
    atomic_dipole = np.reshape(atomic_dipole, (-1, 3))
    assert atomic_dipole.shape[0] == len(sel_ids)
    if full_wannier_spread is not None:
        wannier_spread = np.concatenate(wannier_spread)
    return atomic_dipole, extended_coords, wannier_spread


def build_sel_type_assertion(sel_type, model_path: str, py_cmd="python"):
    return f'''{py_cmd} -c "from deepmd.infer import DeepDipole;dp = DeepDipole({repr(model_path)});assert{repr(sel_type)}==[t for t in dp.tselt]"'''


def dump_dplr_lammps_data(
    fp,
    atoms: Atoms,
    type_map: List[str],
    sel_type: List[int],
    sys_charge_map: List[float],
    model_charge_map: List[float],
):
    """
    dump atoms to LAMMPS data file for DPLR
    the naming convention of params follows DeepMD-kit's

    about dplr: https://docs.deepmodeling.com/projects/deepmd/en/master/model/dplr.html
    about fitting tensor: https://docs.deepmodeling.com/projects/deepmd/en/master/model/train-fitting-tensor.html

    :param fp: file pointer
    :param type_map: the type map of atom type, for example, [O,H]
    :param sel_type: the selected type of atom, for example, [0] means atom type 0, aka O is selected
    :param sys_charge_map: the charge map of atom in system, for example, [6, 1]
    :param model_charge_map: the charge map of atom in model, for example, [-8]
    """

    if len(type_map) != len(sys_charge_map):
        raise ValueError(
            f"type_map {type_map} and sys_charge_map {sys_charge_map} must have the same length"
        )

    if len(sel_type) != len(model_charge_map):
        raise ValueError(
            f"sel_type {sel_type} and model_charge_map {model_charge_map} must have the same length"
        )

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
        charges[new_atoms.symbols == atype] = charge
    # build bonds
    n_bonds = len(r_atom_ids)
    # note: the index of lammps start from 1
    bonds = np.array(
        [
            np.arange(n_bonds) + 1,  # bond id
            [1] * n_bonds,  # bond type
            np.array(r_atom_ids) + 1,  # bond left
            np.array(v_atom_ids) + 1,  # bond right
        ],
        dtype=int,
    ).T

    # write lammps data
    lmp_data = LammpsData(new_atoms)
    lmp_data.charges = charges
    lmp_data.bonds = bonds
    lmp_data.write(fp, atom_style="full", specorder=type_map + vtypes)


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


def get_sel_type(model_path) -> List[int]:
    try:
        from deepmd.infer import DeepDipole
    except ImportError:
        raise ImportError("deepmd-kit is not installed")

    dp = DeepDipole(model_path)
    return [t for t in dp.tselt]


def read_wannier_spread(fname: str):
    """
    Read wannier spread file generated by cp2k

    Parameters
    ----------
    fname : str
        wannier_spread.out file name

    Returns
    -------
    wannier_spread : numpy array
        wannier spread data
    """
    # skip 1, 2, and the last lines
    # save the others as array
    with open(fname, "r", encoding="UTF-8") as f:
        lines = f.readlines()[2:-1]

    wannier = []
    for line in lines:
        # line to array
        line = line.strip().split()
        wannier.append([float(line[1]), float(line[2])])
    return np.array(wannier)


def dplr_v2_to_v3(data_path: str, sel_symbol: list):
    atomic_data_fnames = [
        "atomic_dipole.npy",
        "atomic_polarizability.npy",
        "wannier_spread.npy",
        "atomic_weight.npy",
    ]
    for atomic_data_fname in atomic_data_fnames:
        fnames = glob.glob(
            os.path.join(data_path, "**", atomic_data_fname), recursive=True
        )
        for fname in fnames:
            type_map = np.loadtxt(
                os.path.join(os.path.dirname(fname), "../type_map.raw"),
                dtype=str,
            )
            atype = np.loadtxt(
                os.path.join(os.path.dirname(fname), "../type.raw"),
                dtype=int,
            )
            symbols = np.array(type_map)[atype]
            sel_ids = np.where(np.isin(symbols, sel_symbol))[0]
            n_atoms = len(atype)

            raw_data = np.load(fname)
            n_frames = raw_data.shape[0]
            try:
                raw_data = np.reshape(raw_data, [n_frames, len(sel_ids), -1])
            except ValueError:
                raw_data.reshape([n_frames, n_atoms, -1])
                logger.info(f"Already in v3 format: %s" % fname)
                continue
            n_dim = raw_data.shape[2]

            full_data = np.zeros([n_frames, n_atoms, n_dim])
            full_data[:, sel_ids] = raw_data
            np.save(fname, full_data.reshape([n_frames, -1]))


def dplr_v3_to_v2(data_path: str, sel_symbol: list):
    atomic_data_fnames = [
        "atomic_dipole.npy",
        "atomic_polarizability.npy",
        "wannier_spread.npy",
        "atomic_weight.npy",
    ]
    for atomic_data_fname in atomic_data_fnames:
        fnames = glob.glob(
            os.path.join(data_path, "**", atomic_data_fname), recursive=True
        )
        for fname in fnames:
            type_map = np.loadtxt(
                os.path.join(os.path.dirname(fname), "../type_map.raw"),
                dtype=str,
            )
            atype = np.loadtxt(
                os.path.join(os.path.dirname(fname), "../type.raw"),
                dtype=int,
            )
            symbols = np.array(type_map)[atype]
            sel_ids = np.where(np.isin(symbols, sel_symbol))[0]
            n_atoms = len(atype)

            raw_data = np.load(fname)
            n_frames = raw_data.shape[0]
            try:
                raw_data_reshape = raw_data.reshape([n_frames, n_atoms, -1])
            except ValueError:
                raw_data.reshape([n_frames, len(sel_ids), -1])
                logger.info(f"Already in v2 format: %s" % fname)
                continue
            np.save(fname, raw_data_reshape[:, sel_ids].reshape([n_frames, -1]))
