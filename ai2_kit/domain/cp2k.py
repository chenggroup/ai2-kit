from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.script import BashScript, BashStep, BashTemplate
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.util import merge_dict, dict_nested_get, split_list
from ai2_kit.core.log import get_logger

from typing import List, Union, Tuple
from pydantic import BaseModel
from dataclasses import dataclass

import copy
import os
import io

from .data import LammpsOutputHelper, XyzHelper, Cp2kOutputHelper, ase_atoms_to_cp2k_input_data
from .iface import ICllLabelOutput, BaseCllContext
from .util import loads_cp2k_input, load_cp2k_input, dump_cp2k_input

logger = get_logger(__name__)

class GenericCp2kInputConfig(BaseModel):
    init_system_files: List[str] = []
    limit: int = 50
    input_template: Union[dict, str]
    """
    Input template for cp2k. Could be a dict or content of a cp2k input file.

    Note:
    If you are using files in input templates, it is recommended to use artifact name instead of literal path.
    String starts with '@' will be treated as artifact name.
    For examples, FORCE_EVAL/DFT/BASIS_SET_FILE_NAME = @cp2k/basic_set.
    You can still use literal path, but it is not recommended.
    """

class GenericCp2kContextConfig(BaseModel):
    script_template: BashTemplate
    cp2k_cmd: str = 'cp2k'
    concurrency: int = 5


@dataclass
class GenericCp2kInput:
    config: GenericCp2kInputConfig
    system_files: List[Artifact]
    type_map: List[str]
    initiated: bool = False  # FIXME: this seems to be a bad design idea


@dataclass
class GenericCp2kContext(BaseCllContext):
    config: GenericCp2kContextConfig


@dataclass
class GenericCp2kOutput(ICllLabelOutput):
    cp2k_outputs: List[Artifact]

    def get_labeled_system_dataset(self):
        return self.cp2k_outputs


async def generic_cp2k(input: GenericCp2kInput, ctx: GenericCp2kContext) -> GenericCp2kOutput:
    executor = ctx.resource_manager.default_executor

    # For the first round
    # FIXME: move out from this function, this should be done in the workflow
    if not input.initiated:
        input.system_files += ctx.resource_manager.get_artifacts(input.config.init_system_files)

    # setup workspace
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    [tasks_dir] = executor.setup_workspace(work_dir, ['tasks'])

    # prepare input template
    if isinstance(input.config.input_template, str):
        input_template = loads_cp2k_input(input.config.input_template)
    else:
        input_template = copy.deepcopy(input.config.input_template)

    # resolve data files
    lammps_dump_files: List[Artifact] = []
    xyz_files: List[Artifact] = []

    # TODO: support POSCAR in the future
    # TODO: refactor the way of handling different file formats
    system_files = ctx.resource_manager.resolve_artifacts(input.system_files)
    for system_file in system_files:
        if LammpsOutputHelper.is_match(system_file):
            lammps_out = LammpsOutputHelper(system_file)
            lammps_dump_files.extend(lammps_out.get_selected_dumps())
        elif XyzHelper.is_match(system_file):
            xyz_files.append(system_file)
        else:
            raise ValueError(f'unsupported format {system_file.url}: {system_file.format}')

    # create task dirs and prepare input files
    cp2k_task_dirs = []
    if lammps_dump_files or xyz_files:
        cp2k_task_dirs = executor.run_python_fn(make_cp2k_task_dirs)(
            lammps_dump_files=[a.to_dict() for a in lammps_dump_files],
            xyz_files=[a.to_dict() for a in xyz_files],
            type_map=input.type_map,
            base_dir=tasks_dir,
            input_template=input_template,
            limit= 0 if input.initiated else input.config.limit,  # initialize all data if not initiated
        )
    else:
        logger.warn('no available candidates, skip')
        return GenericCp2kOutput(cp2k_outputs=[])

    # build commands
    steps = []
    for cp2k_task_dir in cp2k_task_dirs:
        steps.append(BashStep(
            cwd=cp2k_task_dir['url'],
            cmd=[ctx.config.cp2k_cmd, '-i input.inp 1>> output 2>> output'],
            checkpoint='cp2k',
        ))

    # submit tasks and wait for completion
    jobs = []
    for i, steps_group in enumerate(split_list(steps, ctx.config.concurrency)):
        if not steps_group:
            continue
        script = BashScript(
            template=ctx.config.script_template,
            steps=steps_group,
        )
        job = executor.submit(script.render(), cwd=tasks_dir,
                              checkpoint_key=f'submit-job/cp2k/{i}:{tasks_dir}')
        jobs.append(job)
    jobs = await gather_jobs(jobs, max_tries=2)

    cp2k_outputs = [Artifact.of(
        url=a['url'],
        format=Cp2kOutputHelper.format,
        executor=executor.name,
        attrs=a['attrs'],
    ) for a in cp2k_task_dirs]

    return GenericCp2kOutput(cp2k_outputs=cp2k_outputs)


def __export_remote_functions():

    def make_cp2k_task_dirs(lammps_dump_files: List[ArtifactDict],
                            xyz_files: List[ArtifactDict],
                            type_map: List[str],
                            input_template: dict,
                            base_dir: str,
                            limit: int = 0,
                            input_file_name: str = 'input.inp',
                            ) -> List[ArtifactDict]:
        """Generate CP2K input files from LAMMPS dump files or XYZ files."""
        import ase.io
        from ase import Atoms

        task_dirs = []
        atoms_list: List[Tuple[ArtifactDict, Atoms]] = []

        # read atoms
        for dump_file in lammps_dump_files:
            atoms_list += [
                (dump_file, atoms)
                for atoms in ase.io.read(dump_file['url'], ':', format='lammps-dump-text', order=False, specorder=type_map)
            ]  # type: ignore
        for xyz_file in xyz_files:
            atoms_list += [
                (xyz_file, atoms)
                for atoms in ase.io.read(xyz_file['url'], ':', format='extxyz')
            ]  # type: ignore

        if limit > 0:
            atoms_list = atoms_list[:limit]

        for i, (file, atoms) in enumerate(atoms_list):
            # create task dir
            task_dir = os.path.join(base_dir, f'{str(i).zfill(6)}')
            os.makedirs(task_dir, exist_ok=True)

            # TODO: should also support input_template
            # find input template in data_file attrs, if not found, use input_template as default
            input_data_file = dict_nested_get(file, ['attrs', 'cp2k', 'input_template_file'],  None)  # type: ignore
            if isinstance(input_data_file, str):
                with open(input_data_file, 'r') as f:
                    input_data = load_cp2k_input(f)
            else:
                input_data = copy.deepcopy(input_template)

            coords, cell = ase_atoms_to_cp2k_input_data(atoms)
            merge_dict(input_data, {
                'FORCE_EVAL': {
                    'SUBSYS': {
                        # FIXME: this is a dirty hack, we should make dump_cp2k_input support COORD
                        'COORD': dict.fromkeys(coords, ''),
                        'CELL': {
                            'A': ' '.join(map(str, cell[0])),
                            'B': ' '.join(map(str, cell[1])),
                            'C': ' '.join(map(str, cell[2])),
                        }
                    }
                }
            })
            with open(os.path.join(task_dir, input_file_name), 'w') as f:
                dump_cp2k_input(input_data, f)

            task_dirs.append({
                'url': task_dir,
                'attrs': file['attrs'],
            })

        return task_dirs

    return make_cp2k_task_dirs

make_cp2k_task_dirs = __export_remote_functions()
