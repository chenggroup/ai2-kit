from ai2_kit.core.executor import Executor
from ai2_kit.core.artifact import Artifact
from ai2_kit.core.script import BashScript, BashStep, BashTemplate
from ai2_kit.core.job import GatherJobsFuture, MapFuture, retry_fn, DummyFuture, IFuture

from ai2_kit.core.util import merge_dict
from ai2_kit.core.log import get_logger


from typing import List, Optional
from pydantic import BaseModel
from dataclasses import dataclass

import copy
import os

from .constant import (
    LAMMPS_TRAJ_DIR,
    LAMMPS_TRAJ_SUFFIX,
)

logger = get_logger(__name__)

class GeneralCp2kInputConfig(BaseModel):
    max_candidates: int = 50
    input_template: dict
    basic_set_file: Optional[str]
    potential_file: Optional[str]

class GeneralCp2kContextConfig(BaseModel):
    script_template: BashTemplate
    cp2k_cmd: str = 'cp2k'
    concurrency: int = 5

@dataclass
class GeneralCp2kInput:
    config: GeneralCp2kInputConfig
    candidates: List[Artifact]
    type_map: List[str]

    basic_set_file: Optional[Artifact] = None
    potential_file: Optional[Artifact] = None

@dataclass
class GeneralCp2kContext:
    config: GeneralCp2kContextConfig
    executor: Executor
    path_prefix: str

@dataclass
class GeneralCp2kOutput:
    dp_data_sets: List[Artifact]
    cp2k_outputs: List[Artifact]

def general_cp2k(input: GeneralCp2kInput, ctx: GeneralCp2kContext) -> IFuture[GeneralCp2kOutput]:
    # setup dirs
    work_dir = ctx.executor.get_full_path(ctx.path_prefix)
    ctx.executor.mkdir(work_dir)
    logger.info('work_dir is %s', work_dir)

    tasks_dir = os.path.join(work_dir, 'tasks')
    ctx.executor.mkdir(tasks_dir)
    logger.info('tasks_dir is %s', tasks_dir)

    dp_data_dir = os.path.join(work_dir, 'output_dp_data')
    ctx.executor.mkdir(dp_data_dir)
    logger.info('dp_data_dir is %s', dp_data_dir)

    # resolve input template
    input_template = copy.deepcopy(input.config.input_template)
    # FIXME: potential_file and basic_set_file should be list
    if input.basic_set_file is not None:
        merge_dict(input_template,  {
            'FORCE_EVAL': {
                'DFT': {
                    'BASIS_SET_FILE_NAME': input.basic_set_file.url,
                }
            }
        })
    if input.potential_file is not None:
        merge_dict(input_template,  {
            'FORCE_EVAL': {
                'DFT': {
                    'POTENTIAL_FILE_NAME': input.potential_file.url,
                }
            }
        })

    # resolve candidate files
    lammps_traj_files = []

    for candidate in input.candidates:
        by = candidate.attrs['by']
        if by == 'lammps':
            traj_files = [os.path.join(candidate.url, LAMMPS_TRAJ_DIR, f'{i}{LAMMPS_TRAJ_SUFFIX}') for i in candidate.attrs['passed']]
            lammps_traj_files.extend(traj_files)
        else:
            logger.warn('skip unsupported data source: %s, generated by: %s', candidate.url, by)
            continue

    # limit max_candidates
    # TODO: allow user to choose random sample
    lammps_traj_files = lammps_traj_files[:input.config.max_candidates]

    # create task dirs and prepare input files
    cp2k_task_dirs = []
    if lammps_traj_files:
        cp2k_task_dirs = [
            os.path.join(tasks_dir, str(i).zfill(6))
            for i in range(len(lammps_traj_files))
        ]
        ctx.executor.run_python_fn(make_cp2k_task_dirs)(
            task_dirs=cp2k_task_dirs,
            traj_files=lammps_traj_files,
            traj_fmt='lammps/dump',
            type_map=input.type_map,
            input_template=input_template,
        )
    else:
        logger.warn('no available candidates')
        return DummyFuture(GeneralCp2kOutput(dp_data_sets=[], cp2k_outputs=[]))

    # run cp2k tasks
    # group tasks by concurrency
    concurrency = ctx.config.concurrency
    steps_group = [list() for _ in range(concurrency)]
    for i, cp2k_task_dir in enumerate(cp2k_task_dirs):
        steps = steps_group[i % concurrency]
        step = BashStep(
            cwd=cp2k_task_dir,
            cmd=[ctx.config.cp2k_cmd, '-i input.inp 1>> output 2>> output'],
            checkpoint='cp2k',
        )
        steps.append(step)

    jobs = []
    for steps in steps_group:
        if not steps:
            continue
        script = BashScript(
            template=ctx.config.script_template,
            steps=steps,
        )
        job = ctx.executor.submit(script.render(), cwd=tasks_dir)
        jobs.append(job)

    future = GatherJobsFuture(jobs, done_fn=retry_fn(max_tries=2), raise_exception=True)

    # TODO: use a common intermediate data format instead of deepmd/npy
    # ref: https://www.reddit.com/r/comp_chem/comments/10q0isd/whats_the_best_data_format_to_transfer/
    def convert_to_dp_data(_):
        ctx.executor.run_python_fn(cp2k_output_to_dp_data)(
            task_dirs=cp2k_task_dirs,
            output_dir=dp_data_dir,
            type_map=input.type_map)
        dp_data_sets =[
            Artifact(
                executor=ctx.executor.name,
                url=dp_data_dir,
                attrs=dict(fmt='deepmd', set_size=len(cp2k_task_dirs))
            ), # type: ignore
        ]
        cp2k_outputs = [
            Artifact(
                executor=ctx.executor.name,
                url=task_dir,
                attrs=dict(fmt='cp2k')
            )  for task_dir in cp2k_task_dirs  # type: ignore
        ]
        return GeneralCp2kOutput(dp_data_sets=dp_data_sets, cp2k_outputs=cp2k_outputs)
    return MapFuture(future, convert_to_dp_data)


def __make_cp2k_task_dirs():
    """cloudpickle compatible: https://stackoverflow.com/questions/75292769"""
    def make_cp2k_task_dirs(traj_files: List[str],
                            task_dirs: List[str],
                            type_map: List[str],
                            traj_fmt: str,
                            input_template: dict,
                            input_file_name: str = 'input.inp',
                            ):
        import dpdata
        import numpy as np
        from cp2k_input_tools import DEFAULT_CP2K_INPUT_XML
        from cp2k_input_tools.generator import CP2KInputGenerator

        assert len(traj_files) == len(task_dirs), 'len(input_files) != len(task_dirs)'

        cp2k_generator = CP2KInputGenerator(DEFAULT_CP2K_INPUT_XML)

        for i in range(len(traj_files)):
            traj_file = traj_files[i]
            task_dir = task_dirs[i]

            os.makedirs(task_dir, exist_ok=True)
            input_data = copy.deepcopy(input_template)
            system = dpdata.System(traj_file, fmt=traj_fmt, type_map=type_map)

            # format coords
            atom_names = np.array(system['atom_names'])  # type: ignore
            atom_types = system['atom_types']  # type: ignore
            coord_arr = system['coords'][0]  # type: ignore
            kind_arr = atom_names[atom_types]  # type: ignore
            coords = [ str(k) + ' ' + ' '.join(str(x) for x in c)  for k, c in zip(kind_arr, coord_arr) ] # type: ignore

            # format cell
            cell = system['cells'][0]
            cell = np.reshape(cell, [3,3])

            # override input_data
            merge_dict(input_data, {
                'FORCE_EVAL': {
                    'SUBSYS': {
                        'COORD': {
                            '*': coords
                        },
                        'CELL': {
                            'A': list(cell[0, :]),
                            'B': list(cell[1, :]),
                            'C': list(cell[2, :]),
                        }
                    }
                }
            })
            input_text = '\n'.join(cp2k_generator.line_iter(input_data))
            with open(os.path.join(task_dir, input_file_name), 'w') as f:
                f.write(input_text)

    return make_cp2k_task_dirs
make_cp2k_task_dirs = __make_cp2k_task_dirs()


def __cp2k_output_to_dp_data():
    """cloudpickle compatible: https://stackoverflow.com/questions/75292769"""
    def cp2k_output_to_dp_data(task_dirs: List[str], output_dir: str, type_map: List[str]):
        import dpdata
        system = None
        for task_dir in task_dirs:
            _system = dpdata.LabeledSystem(os.path.join(task_dir, 'output'), fmt='cp2k/output', type_map=type_map)
            if system is None:
                system = _system
            else:
                system.append(_system)
        assert system, 'system should not be None, check if task_dirs is empty'
        system.to_deepmd_raw(output_dir)  # type: ignore
        system.to_deepmd_npy(output_dir, set_size = len(system))  # type: ignore

    return cp2k_output_to_dp_data
cp2k_output_to_dp_data = __cp2k_output_to_dp_data()