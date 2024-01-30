from ase import Atoms
import ase.io

from typing import List, Optional
from collections import OrderedDict
import shlex
import os

from ai2_kit.tool.dpdata import register_data_types
from ai2_kit.core.util import list_split, expand_globs, cmd_with_checkpoint
from ai2_kit.domain.cp2k import dump_coord_n_cell
from ai2_kit.core.log import get_logger


register_data_types()
logger = get_logger(__name__)


def ensure_file_exists(file_path: str):
    assert os.path.isfile(file_path), f'{file_path} is not a file'
    return file_path


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
        self._cp2k_inputs[tag] = ensure_file_exists(file_path)
        return self

    def make_task(self, out_dir: str):
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

    def make_batch(self,
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


cmd_entry = {
        'cp2k-labeling': Cp2kLabelTaskBuilder,
    }


if __name__ == '__main__':
    from fire import Fire
    Fire(cmd_entry)
