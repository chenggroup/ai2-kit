from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.script import BashScript, BashStep, BashTemplate
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.util import merge_dict, dict_nested_get, list_split, list_sample, dump_json, dump_text
from ai2_kit.core.log import get_logger

from typing import List, Union, Tuple, Literal, Optional
from pydantic import BaseModel
from dataclasses import dataclass
from ase import Atoms

import copy
import os

from .data import DataFormat, ase_atoms_to_cp2k_input_data, artifacts_to_ase_atoms
from .iface import ICllLabelOutput, BaseCllContext
from .util import loads_cp2k_input, load_cp2k_input, dump_cp2k_input

logger = get_logger(__name__)


class CllCp2kInputConfig(BaseModel):
    init_system_files: List[str] = []
    wfn_warmup_template: Union[None, dict, str] = None
    """
    Warmup template for cp2k. Could be a dict or content of a cp2k input file.
    This template will be used to generate input files for warmup runs.
    The warmup runs can be used to generate wave function files for the main runs.
    """
    input_template: Union[dict, str] = dict()
    """
    Input template for cp2k. Could be a dict or content of a cp2k input file.
    """
    limit: int = 50
    """
    Limit of the number of systems to be labeled.
    """
    limit_method: Literal["even", "random", "truncate"] = "even"


class CllCp2kContextConfig(BaseModel):
    script_template: BashTemplate
    cp2k_cmd: str = 'cp2k'
    post_cp2k_cmd: str = 'echo "no post_cp2k_cmd"'
    concurrency: int = 5


@dataclass
class CllCp2kInput:
    config: CllCp2kInputConfig
    system_files: List[Artifact]
    type_map: List[str]
    initiated: bool = False  # FIXME: this seems to be a bad design idea


@dataclass
class CllCp2kContext(BaseCllContext):
    config: CllCp2kContextConfig


@dataclass
class GenericCp2kOutput(ICllLabelOutput):
    cp2k_outputs: List[Artifact]

    def get_labeled_system_dataset(self):
        return self.cp2k_outputs


async def cll_cp2k(input: CllCp2kInput, ctx: CllCp2kContext) -> GenericCp2kOutput:
    executor = ctx.resource_manager.default_executor

    # For the first round
    # FIXME: move out from this function, this should be done in the workflow
    if not input.initiated:
        input.system_files += ctx.resource_manager.resolve_artifacts(input.config.init_system_files)

    if len(input.system_files) == 0:
        return GenericCp2kOutput(cp2k_outputs=[])

    # setup workspace
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    [tasks_dir] = executor.setup_workspace(work_dir, ['tasks'])

    # prepare input template
    if isinstance(input.config.input_template, str):
        input_template = loads_cp2k_input(input.config.input_template)
    else:
        input_template = copy.deepcopy(input.config.input_template)

    # create task dirs and prepare input files
    cp2k_task_dirs = executor.run_python_fn(make_cp2k_task_dirs)(
        system_files=[a.to_dict() for a in input.system_files],
        type_map=input.type_map,
        base_dir=tasks_dir,
        input_template=input_template,
        # initialize all data if not initiated
        limit=0 if not input.initiated else input.config.limit,
        wfn_warmup_template=input.config.wfn_warmup_template,
    )

    # build commands
    steps = []
    for cp2k_task_dir in cp2k_task_dirs:
        # run warmup if needed
        # note: use if-else instead of boolean shortcut to avoid wrong status
        cmd = '\n'.join([
            f'if [ -f wfn_warmup.inp ]; then {ctx.config.cp2k_cmd} -i wfn_warmup.inp &> wfn_warmup.out; fi && \\',
            f'{ctx.config.cp2k_cmd} -i input.inp &> output && {ctx.config.post_cp2k_cmd}',
        ])
        steps.append(BashStep(
            cwd=cp2k_task_dir['url'],
            cmd=cmd,
            checkpoint='cp2k',
        ))

    # submit tasks and wait for completion
    jobs = []
    for i, steps_group in enumerate(list_split(steps, ctx.config.concurrency)):
        if not steps_group:
            continue
        script = BashScript(
            template=ctx.config.script_template,
            steps=steps_group,
        )
        job = executor.submit(script.render(), cwd=tasks_dir,
                              checkpoint_key=f'queue-job:cp2k:{tasks_dir}:{i}')
        jobs.append(job)
    jobs = await gather_jobs(jobs, max_tries=2)

    cp2k_outputs = [Artifact.of(
        url=a['url'],
        format=DataFormat.CP2K_OUTPUT_DIR,
        executor=executor.name,
        attrs=a['attrs'],
    ) for a in cp2k_task_dirs]

    return GenericCp2kOutput(cp2k_outputs=cp2k_outputs)


def __export_remote_functions():

    def make_cp2k_task_dirs(system_files: List[ArtifactDict],
                            type_map: List[str],
                            input_template: dict,
                            base_dir: str,
                            limit: int = 0,
                            sample_method: Literal["even", "random", "truncate"] = "even",
                            input_file_name: str = 'input.inp',
                            wfn_warmup_template: Union[None, dict, str] = None,
                            warmup_file_name: str = 'wfn_warmup.inp'
                            ) -> List[ArtifactDict]:
        """Generate CP2K input files from LAMMPS dump files or XYZ files."""
        task_dirs = []
        atoms_list: List[Tuple[ArtifactDict, Atoms]] = artifacts_to_ase_atoms(system_files, type_map=type_map)

        if limit > 0:
            atoms_list = list_sample(atoms_list, limit, method=sample_method)

        for i, (data_file, atoms) in enumerate(atoms_list):
            # create task dir
            task_dir = os.path.join(base_dir, f'{str(i).zfill(6)}')
            os.makedirs(task_dir, exist_ok=True)
            dump_json(data_file, os.path.join(task_dir, 'debug.data-file.json'))

            # load system-wise config from attrs
            overridable_params: dict = copy.deepcopy(dict_nested_get(data_file, ['attrs', 'cp2k'], dict()))  # type: ignore

            wfn_warmup_template = overridable_params.get('wfn_warmup_template') or copy.deepcopy(wfn_warmup_template)
            input_template = overridable_params.get('input_template') or copy.deepcopy(input_template)

            if wfn_warmup_template:
                with open(os.path.join(task_dir, warmup_file_name), 'w') as f:
                    make_cp2k_input(f, wfn_warmup_template, atoms)

            with open(os.path.join(task_dir, input_file_name), 'w') as f:
                make_cp2k_input(f, input_template, atoms)

            task_dirs.append({
                'url': task_dir,
                'attrs': data_file['attrs'],
            })

        return task_dirs


    def make_cp2k_input(fp, input_template: Union[str, dict], atoms: Atoms):
        if isinstance(input_template, str):
            input_template = loads_cp2k_input(input_template)
        assert isinstance(input_template, dict) and input_template, 'input_data must be a non-empty dict'
        coords, cell = ase_atoms_to_cp2k_input_data(atoms)
        merge_dict(input_template, {
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
        dump_cp2k_input(input_template, fp)

    return (
        make_cp2k_task_dirs,
    )


(
    make_cp2k_task_dirs,
) = __export_remote_functions()
