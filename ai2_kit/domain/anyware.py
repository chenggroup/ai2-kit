from ai2_kit.core.script import BashTemplate, BashStep, BashScript
from ai2_kit.core.artifact import Artifact
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.util import list_split, dump_text
from ai2_kit.core.pydantic import BaseModel
from ai2_kit.core.connector import get_ln_cmd

from typing import List, Optional, Mapping, Any
from dataclasses import dataclass
from string import Template

import os
import itertools
import random
import ase.io


from .data import artifacts_to_ase_atoms_v2, DataFormat
from .iface import BaseCllContext, ICllExploreOutput
from .cp2k import dump_coord_n_cell
from .lammps import _get_dp_models_variables


class AnywareContextConfig(BaseModel):
    script_template: BashTemplate
    concurrency: int = 5


class AnywareConfig(BaseModel):

    system_files: List[str]
    """
    Artifact keys to the system
    """

    template_files: Mapping[str, str]
    """
    Templates files that will generate for each explore tasks,
    You can use $$VAR_NAME to reference the variables defined in product_vars and broadcast_vars.

    Besides, the following build-in variables are also available:
    - SYSTEM_FILE: the path of the system file
    - DP_MODELS: the path of the deep potential models, in the format of '1.pb 2.pb 3.pb 4.pb'

    You can use literal string to define the template file,
    or use !load_text to load the content from a file.

    For example, if you define a template file named 'cp2k.inp' with the following content:
        cp2k-warmup.inp: |
          &GLOBAL
          ...
          &END GLOBAL
        cp2k.inp: !load_text cp2k.inp

    """

    product_vars: Mapping[str, List[str]] = {}
    """
    Define template variables by Cartesian product
    The variable can be referenced in the template file with the following format:
    If there are too many variables, it will generate a large number of tasks,
    in this case, you can use broadcast_vars to reduce the number of tasks.

    $$VAR_NAME
    """

    broadcast_vars: Mapping[str, List[str]] = {}
    """
    Define template variables by broadcast (broadcast as in numpy).
    It's the same as product_vars, except that it will broadcast the variable to all other variables.
    """

    system_file_name: str
    """
    The name of the system file you want to generate,
    for example, 'system.xyz', 'coord_n_cell.inc', etc
    """

    system_file_format: str
    """
    The format of the system file you want to generate,
    for example, `lammps-data`, `cp2k-inc`,  etc

    For all supported data, you can refer to ase.io
    https://wiki.fysik.dtu.dk/ase/ase/io/io.html

    Custom formats:
    - cp2k-inc: coord & cell in the format of CP2K include file, can be used in CP2K input file via `@include coord_n_cell.inc`
    """

    submit_script: str
    """
    A bash script that will be executed in each task directory to submit the task.
    For example,

    mpirun cp2k.popt -i cp2k.inp &> cp2k.out
    """

    post_process_fn: Optional[str] = None
    """
    A python function that will be executed after the task is finished.
    You may use this function to post-process the results.

    The function must named as `post_process_fn` and accept a list of task directories as input.
    The below is an example of merging multiple file into one by keeping only the last line of each file.

    post_process_fn: |
        def post_process_fn(task_dirs):
            import glob
            for task_dir in task_dirs:
                files = glob.glob(os.path.join(task_dir, '*.out'))  # file to merge
                with open(os.path.join(task_dir, 'merged.out'), 'w') as fp:
                    for file in files:
                        with open(file, 'r') as f:
                            lines = f.readlines()
                            if len(lines) > 0:
                                fp.write(lines[-1])
    """

    delimiter: str = '$$'
    """
    delimiter for template
    """

    shuffle: bool = False
    """
    shuffle the combination of system_files, product_vars and broadcast_vars
    """

@dataclass
class AnywareInput:
    config: AnywareConfig
    new_system_files: List[Artifact]
    dp_models: Mapping[str, List[Artifact]]
    type_map: List[str]
    mass_map: List[float]


@dataclass
class AnywareContext(BaseCllContext):
    config: AnywareContextConfig


@dataclass
class AnywareOutput(ICllExploreOutput):
    output_dirs: List[Artifact]

    def get_model_devi_dataset(self) -> List[Artifact]:
        return self.output_dirs


async def anyware(input: AnywareInput, ctx: AnywareContext) -> AnywareOutput:
    executor = ctx.resource_manager.default_executor
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)

    if len(input.new_system_files) > 0:
        data_files = input.new_system_files
    else:
        data_files = ctx.resource_manager.resolve_artifacts(input.config.system_files)
    assert len(data_files) > 0, 'no data files found'

    task_artifacts = executor.run_python_fn(make_anyware_task_dirs)(
        work_dir=work_dir,
        data_files=data_files,
        dp_models={k: [m.url for m in v] for k, v in input.dp_models.items()},
        type_map=input.type_map,
        mass_map=input.mass_map,
        product_vars=input.config.product_vars,
        broadcast_vars=input.config.broadcast_vars,
        template_files=input.config.template_files,
        template_delimiter=input.config.delimiter,
        system_file_name=input.config.system_file_name,
        system_file_format=input.config.system_file_format,
        shuffle=input.config.shuffle,
    )
    steps = []
    for task_artifact in task_artifacts:
        steps.append(BashStep(
            cwd=task_artifact.url, cmd=input.config.submit_script, checkpoint='submit')
        )

    # # submit jobs by the number of concurrency
    jobs = []
    for i, steps_group in enumerate(list_split(steps, ctx.config.concurrency)):
        if not steps_group:
            continue
        script = BashScript(
            template=ctx.config.script_template,
            steps=steps_group,
        )
        job = executor.submit(script.render(), cwd=work_dir)
        jobs.append(job)

    await gather_jobs(jobs, max_tries=2)

    if input.config.post_process_fn:
        executor.run_python_fn(run_post_process_fn)(
            post_process_fn=input.config.post_process_fn,
            task_dirs=[task.url for task in task_artifacts]
        )

    return AnywareOutput(output_dirs=task_artifacts)


def run_post_process_fn(post_process_fn: str, task_dirs: List[str]):
    _locals = {}
    exec(post_process_fn, None, _locals)
    _locals['post_process_fn'](task_dirs)


def make_anyware_task_dirs(work_dir: str,
                           data_files: List[Artifact],
                           dp_models: Mapping[str, List[str]],
                           type_map: List[str],
                           mass_map: List[float],
                           product_vars: Mapping[str, List[str]],
                           broadcast_vars: Mapping[str, List[str]],
                           template_files: Mapping[str, str],
                           template_delimiter: str,
                           system_file_name: str,
                           system_file_format: str,
                           shuffle: bool,
                           ):
    class _Template(Template):
        delimiter = template_delimiter

    # handle data files
    systems_dir = os.path.join(work_dir, 'systems')
    os.makedirs(systems_dir, exist_ok=True)
    atoms_list = []
    atoms_list = artifacts_to_ase_atoms_v2(data_files)

    system_artifacts = []
    for i, (artifact, atoms) in enumerate(atoms_list):
        ancestor = artifact.attrs['ancestor']
        data_file = os.path.join(systems_dir, f'{ancestor}-{i:06d}-{system_file_name}')
        if system_file_format == 'cp2k-inc':
            with open(data_file, 'w') as fp:
                dump_coord_n_cell(fp, atoms)
        elif system_file_format == 'lammps-data':
            ase.io.write(data_file, atoms, format=system_file_format, specorder=type_map)  # type: ignore
        else:
            ase.io.write(data_file, atoms, format=system_file_format)  # type: ignore
        system_artifacts.append(Artifact(url=data_file, attrs=artifact.attrs))
    if shuffle:
        random.shuffle(system_artifacts)

    # handle task dirs
    combination_fields: List[str] = ['SYSTEM_FILE']
    combination_values: List[List[Any]] = [system_artifacts]

    for k, v in product_vars.items():
        combination_fields.append(k)
        if shuffle:
            random.shuffle(v)
        combination_values.append(v)

    combinations = itertools.product(*combination_values)
    combinations = list(map(list, combinations))

    combination_fields.extend(broadcast_vars.keys())
    for i, combination in enumerate(combinations):
        for _vars in broadcast_vars.values():
            combination.append(_vars[i % len(_vars)])

    task_artifacts = []
    tasks_base_dir = os.path.join(work_dir, 'tasks')
    for i, combination in enumerate(combinations):
        task_dir = os.path.join(tasks_base_dir, f'{i:06d}')
        os.makedirs(task_dir, exist_ok=True)
        template_vars = dict(zip(combination_fields, combination))

        # link system_file to task_dir
        system_artifact: Artifact = template_vars.pop('SYSTEM_FILE')
        system_file = os.path.join(task_dir, system_file_name)
        # the reason of not using os.symlink is that it will raise an error if the link already exists
        os.system(get_ln_cmd(system_artifact.url, system_file))
        template_vars['SYSTEM_FILE'] = system_file_name
        # dp models variables
        dp_vars = _get_dp_models_variables(dp_models)
        # generate template files
        for k, v in template_files.items():
            template_file_path = os.path.join(task_dir, k)
            dump_text(_Template(v).substitute(**template_vars, **dp_vars), template_file_path, encoding='utf-8')
        task_artifacts.append(Artifact(url=task_dir,
                                       attrs={**system_artifact.attrs, 'source': system_artifact.url},
                                       format=DataFormat.ANYWARE_OUTPUT_DIR))
    return task_artifacts
