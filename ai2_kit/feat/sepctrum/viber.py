from ase import Atoms
import ase.io
import dpdata

from typing import List, Optional
import shlex
import os

import ai2_kit.tool.dpdata  # register custom fields
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

        self._dipole = None
        self._polar_x = None
        self._polar_y = None
        self._polar_z = None

        self._task_dirs = []

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

    def set_cp2k_inp(self, dipole: str, polar_x: str, polar_y: str, polar_z: str):
        """
        Set cp2k input files for dipole and polarizability calculation.
        The input files should use `@include coord_n_cell.inc` to include the system file.

        :param dipole: cp2k input file for dipole calculation
        :param polar_x: cp2k input file for polarizability calculation along x
        :param polar_y: cp2k input file for polarizability calculation along y
        :param polar_z: cp2k input file for polarizability calculation along z
        """
        self._dipole = dipole
        self._polar_x = polar_x
        self._polar_y = polar_y
        self._polar_z = polar_z

    def make_dirs(self, out_dir: str):
        """
        Make task dirs for cp2k labeling (prepare systems and cp2k input files)

        :param out_dir: output dir
        """
        assert self._atoms_list, 'No system files added'

        task_dirs = []
        for i, atoms in enumerate(self._atoms_list):
            task_name = f'{i:06d}-{atoms.get_chemical_formula()}'
            task_dir = os.path.join(out_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            # dump coord_n_cell.inc
            cp2k_sys_file = os.path.join(task_dir, 'coord_n_cell.inc')
            with open(cp2k_sys_file, 'w') as f:
                dump_coord_n_cell(f, atoms)
            # dump system.xyz for cp2k
            xyz_sys_file = os.path.join(task_dir, 'system.xyz')
            ase.io.write(xyz_sys_file, atoms, format='extxyz', append=True)
            # link cp2k input files to tasks dirs
            for src, target in zip([self._dipole, self._polar_x, self._polar_y, self._polar_z],
                                   ['dipole.inp', 'polar_x.inp', 'polar_y.inp', 'polar_z.inp']):
                assert src is not None, f'{target} is not set'
                src_path = os.path.abspath(src)
                target_path = os.path.join(task_dir, target)
                os.system(f'ln -sf {src_path} {target_path}')
            task_dirs.append(task_dir)

    def make_batch(self,
                   prefix: str = 'batch-{i:02d}.sh',
                   concurrency: int = 5,
                   template: Optional[str] = None,
                   suppress_error: bool = False,
                   cmd: str = 'cp2k.psmp'):
        """
        Make batch script for cp2k labeling

        :param prefix: prefix of batch script
        :param concurrency: concurrency of tasks
        :param template: template of batch script
        """
        assert self._atoms_list, 'No system files added, please call add_system first'
        assert self._task_dirs, 'No task dirs generated, please call make_dirs first'
        if template is None:
            template = '#!/bin/bash'
        else:
            with open(template, 'r') as f:
                template = f.read()

        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        for i, groups in enumerate(list_split(self._task_dirs, concurrency)):
            batch = [template]
            if not suppress_error:
                batch.append('set -e')

            batch.append("for _WORK_DIR_ in \\")
            # use shell for loop to run tasks
            for task_dir in groups:
                batch.append(f'  {shlex.quote(task_dir)} \\')
            batch.extend([
                f'  ; do',
                f'    pushd $_WORK_DIR_',
                cmd_with_checkpoint(f'{cmd} -i dipole.inp  &> dipole.out', 'dipole.done'),
                cmd_with_checkpoint(f'{cmd} -i polar_x.inp &> polar_x.out', 'polar_x.done'),
                cmd_with_checkpoint(f'{cmd} -i polar_y.inp &> polar_y.out', 'polar_y.done'),
                cmd_with_checkpoint(f'{cmd} -i polar_z.inp &> polar_z.out', 'polar_z.done'),
                f'    popd',
                f'  done'
            ])

            batch_file = prefix.format(i=i)
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(batch))

    def _ase_read(self, filename: str, **kwargs):
        kwargs.setdefault('index', ':')
        data = ase.io.read(filename, **kwargs)
        if not isinstance(data, list):
            data = [data]
        self._atoms_list += data



def dpdata_read_cp2k_data(cp2k_dir: str,
                          cp2k_out: str = 'output',
                          wannier: str = 'wannier.xyz',
                          wannier_x: str = 'wannier_x.xyz',
                          wannier_y: str = 'wannier_y.xyz',
                          wannier_z: str = 'wannier_z.xyz',):
    dp_sys = dpdata.LabeledSystem(os.path.join(cp2k_dir, cp2k_out) , fmt='cp2k/output')
    nframes = dp_sys.get_nframes()
    natoms = dp_sys.get_natoms()

    wannier_atoms = ase.io.read(os.path.join(cp2k_dir, wannier), index=0, format='extxyz')
    wannier_atoms_x = ase.io.read(os.path.join(cp2k_dir, wannier_x), index=0, format='extxyz')
    wannier_atoms_y = ase.io.read(os.path.join(cp2k_dir, wannier_y), index=0, format='extxyz')
    wannier_atoms_z = ase.io.read(os.path.join(cp2k_dir, wannier_z), index=0, format='extxyz')


    # build the data of atomic_dipole and atomic_polarizability with numpy


    # assign data to dp_sys
    dp_sys.data['atomic_dipole'] = None  # TODO
    dp_sys.data['atomic_polarizability'] = None  # TODO
    return dp_sys


if __name__ == '__main__':
    dp_sys = dpdata_read_cp2k_data('cp2k_dir',
                                   wannier='al2o3_wannier.xyz',
                                   wannier_x='polar_x.wannier',
                                   wannier_y='polar_y.wannier',
                                   wannier_z='polar_z.wannier',
                                   )
    dp_sys.to_deepmd_npy('test_deepmd')
