from dpdata.unit import econvs
from typing import List, Optional

import numpy as np

import dpdata
import os
import re


from .dplr import set_dplr_ext_from_cp2k_output, get_sel_ids
from ai2_kit.core.log import get_logger

logger = get_logger(__name__)


def dpdata_read_cp2k_dpff_data(
    cp2k_dir: str,
    cp2k_output: str,
    wannier_file: str,
    type_map: List[str],
    sys_charge_map: List[int],
    model_charge_map: List[int],
    ewald_h: float,
    ewald_beta: float,
    ext_efield,
    sel_type: List[int],
    wannier_cutoff: float = 1.0,
    wannier_spread_file: Optional[str] = None,
):
    """
    Gnereate dpdata from cp2k output and wannier file for DPLR

    :param cp2k_dir: the directory of cp2k output and wannier file
    :param cp2k_output: the cp2k output file
    :param wannier_file: the wannier file
    :param type_map: the type map of atom type, for example, [O,H]
    :param sys_charge_map: the charge map of atom in system, for example, [6, 1]
    :param model_charge_map: the charge map of wannier in model, for example, [-8]
    :param ewald_h: the ewald_h parameter used in dplr/dpff model
    :param ewald_beta: the ewald_beta parameter used in dplr/dpff model
    :param ext_efield: the external electric field
    :param sel_type: the selected type of atom, for example, [0] means atom type 0, aka O is selected
    :param wannier_cutoff: the cutoff to allocate wannier centers around atoms
    :param wannier_spread_file: the wannier spread file, if provided, the spread data will be added to dp_sys

    :return dp_sys: dpdata.LabeledSystem
        In addition to the common energy data, the following data are added:
        - atomic_dipole: the atomic dipole of selected atoms
        - ext_efield: the external electric field
        - efield: the direction vector of atomic electric field
        - aparam: the atomic electric field magnitude
    """
    cp2k_output = os.path.join(cp2k_dir, cp2k_output)
    wannier_file = os.path.join(cp2k_dir, wannier_file)
    wannier_spread_file = (
        os.path.join(cp2k_dir, wannier_spread_file) if wannier_spread_file else None
    )
    dp_sys = dpdata.LabeledSystem(cp2k_output, fmt="cp2k/output")
    try:
        dp_sys = set_dpff_ext_from_cp2k_output(
            dp_sys,
            cp2k_output,
            wannier_file,
            type_map,
            sys_charge_map,
            model_charge_map,
            ewald_h,
            ewald_beta,
            ext_efield,
            sel_type,
            wannier_cutoff,
            wannier_spread_file,
        )
    except ValueError:
        dp_sys = None

    return dp_sys


def set_dpff_ext_from_cp2k_output(
    dp_sys: dpdata.LabeledSystem,
    cp2k_output: str,
    wannier_file: str,
    type_map: List[str],
    sys_charge_map: List[int],
    model_charge_map: List[int],
    ewald_h: float,
    ewald_beta: float,
    ext_efield,
    sel_type: List[int],
    wannier_cutoff: float = 1.0,
    wannier_spread_file: Optional[str] = None,
):
    # with atomic_dipole
    # update energy
    # add ext_efield, efield, aparam
    dplr_dp_sys = set_dplr_ext_from_cp2k_output(
        dp_sys,
        wannier_file,
        type_map,
        sel_type,
        wannier_cutoff,
        wannier_spread_file,
        model_charge_map,
    )
    atomic_dipole = dplr_dp_sys.data["atomic_dipole"].reshape(-1, 3)  # type: ignore

    with open(cp2k_output, "r", encoding="UTF-8") as fp:
        raw_energy = get_cp2k_output_total_energy(fp)

    ext_efield = np.reshape(ext_efield, [1, 3])
    natoms = dp_sys.get_natoms()
    nframes = dp_sys.get_nframes()
    if nframes == 0:
        return None
    assert nframes == 1, "Only support one frame"

    symbols = np.array(dp_sys.data["atom_names"])[dp_sys.data["atom_types"]]
    sel_ids = get_sel_ids(dp_sys, type_map, sel_type)

    ion_coords = dp_sys.data["coords"].reshape(-1, 3)
    # atomic_dipole.shape = (natoms * 3)
    wannier_coords = (ion_coords + atomic_dipole)[sel_ids].reshape(-1, 3)
    extended_coords = np.concatenate((ion_coords, wannier_coords), axis=0)

    # get extended charges
    extended_charges = np.zeros(len(extended_coords))
    for symbol, ion_charge in zip(type_map, sys_charge_map):
        extended_charges[np.where(symbols == symbol)[0]] = ion_charge
    sel_symbols = symbols[sel_ids]
    for ii, wannier_charge in zip(sel_type, model_charge_map):
        sel_symbol = type_map[ii]
        extended_charges[np.where(sel_symbols == sel_symbol)[0] + natoms] = (
            wannier_charge
        )
    total_dipole = np.sum(extended_coords * extended_charges.reshape(-1, 1), axis=0)
    energy = raw_energy - np.inner(total_dipole, ext_efield.reshape(3))
    dp_sys.data["energies"] = np.reshape(energy, [-1])
    pbc_efield = get_pbc_atomic_efield(
        dp_sys, extended_coords, extended_charges, ewald_h, ewald_beta
    )
    # natoms * 3
    _efield = pbc_efield + ext_efield
    aparam = np.linalg.norm(_efield, axis=-1)
    efield = np.array(_efield) / aparam.reshape(natoms, 1)

    dp_sys.data["ext_efield"] = ext_efield.reshape(nframes, 3)
    dp_sys.data["efield"] = efield.reshape(nframes, natoms, 3)
    dp_sys.data["aparam"] = aparam.reshape(nframes, natoms, -1)
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
        line = line.strip("\n")
        if start_pattern.match(line):
            flag = True
        elif flag and end_pattern.match(line):
            flag = False
            nloop += 1

        if flag is True:
            data_lines.append(line)
    data_lines = np.reshape(data_lines, (nloop, -1))

    tot_e = 0.0
    # grep the data from the last iteration
    for kw in data_lines[-1, 2:-1]:
        kw = kw.split(":")
        k = kw[0].strip(" ")
        v = float(kw[1]) * econvs["hartree"]
        if k != "Fermi energy":
            tot_e += v
    return tot_e


def get_pbc_atomic_efield(
    dp_sys,
    extended_coords,
    extended_charges,
    ewald_h,
    ewald_beta,
):
    from deepmd.infer.ewald_recp import EwaldRecp

    er = EwaldRecp(ewald_h, ewald_beta)
    natoms = dp_sys.get_natoms()
    cell = dp_sys.data["cells"]
    e, f, v = er.eval(
        extended_coords.reshape(1, -1),
        extended_charges.reshape(1, -1),
        cell.reshape(1, -1),
    )
    extended_efield = f.reshape(-1, 3) / extended_charges.reshape(-1, 1)
    # natoms * 3
    pbc_efield = extended_efield[:natoms]
    return pbc_efield
