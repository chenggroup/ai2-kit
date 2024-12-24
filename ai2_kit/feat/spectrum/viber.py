from typing import List, Dict, Optional
from collections import OrderedDict
from ase import Atoms
import numpy as np
import ase.io
import dpdata
import shlex
import os

from MDAnalysis.lib.distances import distance_array, minimize_vectors

from ai2_kit.core.util import list_split, expand_globs, cmd_with_checkpoint
from ai2_kit.domain.cp2k import dump_coord_n_cell
from ai2_kit.core.log import get_logger

logger = get_logger(__name__)


class Cp2kLabelTaskBuilder:
    """
    This a builder class to generate cp2k labeling task.
    """
    def __init__(self):
        self._atoms_list: List[Atoms] = []
        self._task_dirs: List[str] = []
        self._cp2k_inputs: OrderedDict = OrderedDict()

    def add_system(self, *file_path_or_glob: str, **kwargs):
        """
        Add system files to label

        :param file_path_or_glob: system files or glob pattern
        :param kwargs: kwargs for ase.io.read
        """
        files = expand_globs(file_path_or_glob)
        if len(files) == 0:
            raise FileNotFoundError(f'No file found for {file_path_or_glob}')
        for file in files:
            self._ase_read(file, **kwargs)
        return self

    def add_cp2k_input(self, file_path: str, tag: str):
        """
        Add cp2k input file for dipole or polarizability calculation
        """
        assert tag not in self._cp2k_inputs, f'Tag {tag} already exists'
        self._cp2k_inputs[tag] = _ensure_file_exists(file_path)
        return self

    def make_tasks(self, out_dir: str):
        """
        Make task dirs for cp2k labeling (prepare systems and cp2k input files)
        :param out_dir: output dir
        """
        assert self._atoms_list, 'No system files added'
        assert not self._task_dirs, 'Task dirs already generated'

        task_dirs = []
        for i, atoms in enumerate(self._atoms_list):
            task_name = f'{i:06d}'
            task_dir = os.path.join(out_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            # dump coord_n_cell.inc, so that we can use @include to read system
            cp2k_sys_file = os.path.join(task_dir, 'coord_n_cell.inc')
            with open(cp2k_sys_file, 'w') as f:
                dump_coord_n_cell(f, atoms)
            # dump system.xyz for cp2k,
            # so that use can use COORD_FILE_NAME to read system
            xyz_sys_file = os.path.join(task_dir, 'system.xyz')
            ase.io.write(xyz_sys_file, atoms, format='extxyz', append=True)
            # link cp2k input files to tasks dirs
            for tag, file_path in self._cp2k_inputs.items():
                link_path = os.path.join(task_dir, tag + '.inp')
                os.system(f'ln -sf {os.path.abspath(file_path)} {link_path}')
            task_dirs.append(task_dir)
        self._task_dirs = task_dirs
        logger.info(f'Generated {len(task_dirs)} task dirs')
        return self

    def make_scripts(self,
                     prefix: str = 'cp2k-batch-{i:02d}.sub',
                     concurrency: int = 5,
                     template: Optional[str] = None,
                     ignore_error: bool = False,
                     cp2k_cmd: str = 'cp2k.psmp'):
        """
        Make batch script for cp2k labeling

        :param prefix: prefix of batch script
        :param concurrency: concurrency of tasks
        :param template: template of batch script
        :param ignore_error: ignore error
        :param cmd: command to run cp2k
        """
        assert self._atoms_list, 'No system files added, please call add_system first'
        assert self._task_dirs, 'No task dirs generated, please call make_dirs first'
        if template is None:
            template = '#!/bin/bash'
        else:
            with open(template, 'r') as f:
                template = f.read()
        template += '\n[ -n "$PBS_O_WORKDIR" ] && cd $PBS_O_WORKDIR]\n'

        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        for i, task_group in enumerate(list_split(self._task_dirs, concurrency)):
            if not task_group:
                continue
            batch = [template]
            batch.append("for _WORK_DIR_ in \\")
            # use shell for loop to run tasks
            for task_dir in task_group:
                batch.append(f'  {shlex.quote(task_dir)} \\')
            batch.extend([
                f'; do',
                f'pushd $_WORK_DIR_ || exit 1',
                '#' * 80,
                *[
                    cmd_with_checkpoint(f'{cp2k_cmd} -i {tag}.inp &> {tag}.out', f'{tag}.done', ignore_error)
                    for tag in self._cp2k_inputs.keys()
                ],
                '#' * 80,
                f'popd',
                f'done'
            ])
            batch_file = prefix.format(i=i)
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(batch))
            logger.info(f'Write batch script to {batch_file}')
        return self

    def done(self):
        """
        Mark the end of commands chain
        """

    def _ase_read(self, filename: str, **kwargs):
        kwargs.setdefault('index', ':')
        data = ase.io.read(filename, **kwargs)
        if not isinstance(data, list):
            data = [data]
        self._atoms_list += data
        logger.info(f'Loaded {len(data)} systems from {filename}, total {len(self._atoms_list)} systems')


def dpdata_read_cp2k_viber_data(data_dir: str,
                                lumped_dict: Dict[str, int],
                                output_file: str = 'output',
                                wannier: str = 'wannier.xyz',
                                wannier_x: str = 'wannier_x.xyz',
                                wannier_y: str = 'wannier_y.xyz',
                                wannier_z: str = 'wannier_z.xyz',
                                wacent_symbol='X',
                                cutoff = 1.2,
                                eps = 1e-3,
                                mode = 'both'
                                ):

    """
    read the cp2k output file and wannier functions

    :param data_dir: the directory of the data
    :param lumped_dict: the dictionary of the element and the expected coordination number, e.g. {"O": 4} (for water molecule)
    :param output_file: the cp2k output file name
    :param wannier: the wannier function file name
    :param wannier_x: the wannier function file name of x direction electric field
    :param wannier_y: the wannier function file name of y direction electric field
    :param wannier_z: the wannier function file name of z direction electric field
    :param wacent_symbol: the symbol of the wannier function centroid
    """

    dp_sys = dpdata.LabeledSystem(os.path.join(data_dir, output_file) , fmt='cp2k/output')

    # get cell
    cell = dp_sys.data['cells'][0]

    # get selected atoms ids
    symbols = np.array(dp_sys.data["atom_names"])[dp_sys.data["atom_types"]]
    sel_ids = [np.where(symbols == atype)[0] for atype in lumped_dict.keys()]
    sel_ids = np.concatenate(sel_ids)
    
    # build the data of atomic_dipole and atomic_polarizability with numpy
    wannier_atoms = ase.io.read(os.path.join(data_dir, wannier), index=":", format='extxyz')
    n_frames = len(wannier_atoms)
    n_atoms = np.sum(np.logical_not(wannier_atoms[0].symbols == wacent_symbol))

    lumped_dict_c = lumped_dict.copy()
    del_list = []
    for k in lumped_dict_c.keys():
        if k not in wannier_atoms[0].get_chemical_symbols():
            del_list.append(k)

    for i in del_list:
        del lumped_dict_c[i]

    stc_list = _set_cells(wannier_atoms, cell)  # type: ignore
    wfc_compute_polar = _set_lumped_wfc(stc_list, lumped_dict_c, cutoff, wacent_symbol, to_polar = True)
    wfc_save = _set_lumped_wfc(stc_list, lumped_dict_c, cutoff, wacent_symbol, to_polar = False)

    wannier = np.zeros([n_frames, n_atoms, 3])
    wannier[:, sel_ids, :] = np.reshape(wfc_save, [n_frames, -1, 3])
    dp_sys.data["atomic_dipole"] = wannier

    if mode == 'both':
        wannier_atoms_x = ase.io.read(os.path.join(data_dir, wannier_x), index=":", format='extxyz')
        stc_list = _set_cells(wannier_atoms_x, cell)  # type: ignore
        wfc_x = _set_lumped_wfc(stc_list, lumped_dict_c, cutoff, wacent_symbol, to_polar = True)

        wannier_atoms_y = ase.io.read(os.path.join(data_dir, wannier_y), index=":", format='extxyz')
        stc_list = _set_cells(wannier_atoms_y, cell)  # type: ignore
        wfc_y = _set_lumped_wfc(stc_list, lumped_dict_c, cutoff, wacent_symbol, to_polar = True)

        wannier_atoms_z = ase.io.read(os.path.join(data_dir, wannier_z), index=":", format='extxyz')
        stc_list = _set_cells(wannier_atoms_z, cell)  # type: ignore
        wfc_z = _set_lumped_wfc(stc_list, lumped_dict_c, cutoff, wacent_symbol, to_polar = True)

        polar = np.zeros((wfc_compute_polar.shape[0], wfc_compute_polar.shape[1], 3), dtype = float)

        polar[:, :, 0] = (wfc_x - wfc_compute_polar) / eps
        polar[:, :, 1] = (wfc_y - wfc_compute_polar) / eps
        polar[:, :, 2] = (wfc_z - wfc_compute_polar) / eps

        dp_sys.data['atomic_polarizability'] = np.reshape(polar, [n_frames, n_atoms, 9])
    elif mode == 'dipole_only':
        dp_sys.data['atomic_polarizability'] = np.full((n_frames, n_atoms, 9), -1.0)
    else:
        logger.warning(f"There is no mode called '{mode}', expected 'both' or 'dipole_only'")

    return dp_sys


def _get_lumped_wacent_poses_rel(stc: Atoms, elem_symbol, wacent_symbol, cutoff=1.2, expected_cn=4):
    """
    determine the positions of the wannier centers around O
    and sum it into the wannier centroid.
    """
    elem_idx = np.where(stc.symbols == elem_symbol)[0]
    wacent_idx = np.where(stc.symbols == wacent_symbol)[0]
    elem_poses = stc.positions[elem_idx]
    wacent_poses = stc.positions[wacent_idx]

    cellpar = stc.cell.cellpar()
    assert cellpar is not None, "cellpar is None"
    #dist_matrix
    dist_mat = distance_array(elem_poses, wacent_poses, box=cellpar)

    #each row get distance and select the candidates
    lumped_wacent_poses_rel = []
    for elem_entry, dist_vec in enumerate(dist_mat):
        if cutoff == None:
            mindist_index = np.argpartition(np.array(dist_vec), elem_symbol)[elem_symbol:]
            bool_vec = np.full(dist_vec.shape, False)
            bool_vec[mindist_index] = True
        else:
            bool_vec = (dist_vec < cutoff)
        cn = np.sum(bool_vec)

        # modify neighbor wannier centers coords relative to the center element atom
        neig_wacent_poses = wacent_poses[bool_vec, :]
        neig_wacent_poses_rel = neig_wacent_poses - elem_poses[elem_entry]
        neig_wacent_poses_rel = minimize_vectors(neig_wacent_poses_rel, box=cellpar)
        lumped_wacent_pos_rel = neig_wacent_poses_rel.mean(axis=0)

        if cn != expected_cn:
            logger.warning(f"Coordination number of {elem_symbol} is {cn}, expected {expected_cn}")

        lumped_wacent_poses_rel.append(lumped_wacent_pos_rel)

    lumped_wacent_poses_rel = np.stack(lumped_wacent_poses_rel)
    return lumped_wacent_poses_rel

def _is_X(item):
    if not isinstance(item,str): 
        return False
    if item == 'X': 
        return True
    return False

def _can_retain(item,elem_symbol):
    if not isinstance(item, str):
        return True
    if item != elem_symbol:
        return True
    return False

def _set_lumped_wfc(stc_list, lumped_dict, cutoff, wacent_symbol, to_polar = True):
    """
    set the wannier function centroids
    """
    X_pos = []
    if to_polar:
        for stc in stc_list:
            x_symbol = list(stc.symbols)

            for elem_symbol, expected_cn in lumped_dict.items():
                lumped_wacent_poses_rel = _get_lumped_wacent_poses_rel(
                    stc=stc, elem_symbol=elem_symbol, wacent_symbol = wacent_symbol,
                    cutoff = cutoff, expected_cn=expected_cn)
                out_elem_symbol = list(lumped_wacent_poses_rel)
                x_symbol = [item if _can_retain(item,elem_symbol) else out_elem_symbol.pop(0) for item in x_symbol]

            x_symbol = [item for item in x_symbol if not _is_X(item)]
            x_symbol = [np.array([0.,0.,0.]) if isinstance(item, str) else item for item in x_symbol]
            x_symbol = np.concatenate(x_symbol ,axis = 0)
            X_pos.append(np.array(x_symbol))
        wfc_pos = np.array(X_pos)
        return wfc_pos
    else:
        for stc in stc_list:
            for elem_symbol, expected_cn in lumped_dict.items():
                lumped_wacent_poses_rel = _get_lumped_wacent_poses_rel(
                    stc=stc, elem_symbol=elem_symbol, wacent_symbol = wacent_symbol,
                    cutoff = cutoff, expected_cn=expected_cn)
                X_pos.append(np.reshape(lumped_wacent_poses_rel, (len(stc_list), -1)))
        wfc_pos = np.concatenate(X_pos,axis = 1)
        return wfc_pos

def _set_cells(stc_list: List[Atoms], cell):
    for stc in stc_list:
        stc.set_cell(cell)
        stc.set_pbc(True)
    return stc_list


def _ensure_file_exists(file_path: str):
    assert os.path.isfile(file_path), f'{file_path} is not a file'
    return file_path


cmd_entry = {
    'cp2k-labeling': Cp2kLabelTaskBuilder,
}
