from ai2_kit.core.script import BashTemplate, BashStep, BashScript
from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.log import get_logger
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.util import list_split, merge_dict
from ai2_kit.core.pydantic import BaseModel

from typing import List, Optional
from dataclasses import dataclass
from string import Template

import numpy as np
import ase.io
import copy
import os
import re

from .iface import BaseCllContext, ICllExploreOutput
from .constant import DEFAULT_LASP_IN, DEFAULT_LAMMPS_TEMPLATE_FOR_DP_SSW, MODEL_DEVI_OUT
from .data import artifacts_to_ase_atoms, DataFormat


logger = get_logger(__name__)


class CllLaspInputConfig(BaseModel):
    class Potential(BaseModel):
        class LammpsPotential(BaseModel):
            input_template: str = DEFAULT_LAMMPS_TEMPLATE_FOR_DP_SSW

        lammps: Optional[LammpsPotential] = None  # currently only lammps is supported

    input_template: dict
    """
    Input template for LASP
    """
    potential: Potential

    system_files: List[str]
    """
    Initial system files to explore
    """


class CllLaspContextConfig(BaseModel):
    lasp_cmd: str = 'lasp'
    script_template: BashTemplate
    concurrency: int = 5


@dataclass
class CllLaspInput:
    config: CllLaspInputConfig
    type_map: List[str]
    mass_map: List[float]
    models: List[Artifact]
    new_system_files: List[Artifact]


@dataclass
class CllLaspContext(BaseCllContext):
    config: CllLaspContextConfig


@dataclass
class CllLaspOutput(ICllExploreOutput):
    output_dirs: List[Artifact]

    def get_model_devi_dataset(self) -> List[Artifact]:
        return self.output_dirs


async def cll_lasp(input: CllLaspInput, ctx: CllLaspContext):
    executor = ctx.resource_manager.default_executor

    # setup workspace
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    tasks_dir, = executor.setup_workspace(work_dir, ['tasks'])

    # resolve artifacts
    if len(input.new_system_files) > 0:
        systems = input.new_system_files
    else:
        systems = ctx.resource_manager.resolve_artifacts(input.config.system_files)

    # setup configuration
    lasp_in_data = copy.deepcopy(DEFAULT_LASP_IN)

    lammps_input_template = None
    if input.config.potential.lammps is not None:
        lasp_in_data['potential'] = 'lammps'
        lammps_input_template = input.config.potential.lammps.input_template
    else:
        raise ValueError('At least one potential should be specified.')

    lasp_in_data = merge_dict(lasp_in_data, input.config.input_template)
    lasp_in_data['explore_type'] = 'ssw'
    lasp_in_data['SSW.output'] = 'T'

    # make task dirs
    task_dirs = executor.run_python_fn(make_lasp_task_dirs)(
        systems=[a.to_dict() for a in systems],
        type_map=input.type_map, mass_map=input.mass_map,
        lasp_in_data=lasp_in_data,
        dp_models=[m.url for m in input.models],
        lammps_input_template=lammps_input_template,
        base_dir=tasks_dir,
    )

    # generate steps
    lasp_cmd = f'{ctx.config.lasp_cmd} 1 > lasp.out > lasp.err'
    steps = []
    for task_dir in task_dirs :
        steps.append(BashStep(cwd=task_dir['url'], cmd=lasp_cmd, checkpoint='lasp'))

    # submit jobs by the number of concurrency
    jobs = []
    for i, steps_group in enumerate(list_split(steps, ctx.config.concurrency)):
        if not steps_group:
            continue
        script = BashScript(
            template=ctx.config.script_template,
            steps=steps_group,
        )
        job = executor.submit(script.render(), cwd=tasks_dir)
        jobs.append(job)
    await gather_jobs(jobs, max_tries=2)

    # process outputs
    executor.run_python_fn(process_lasp_outputs)(task_dirs=[a['url'] for a in task_dirs])

    output_dirs = [
        Artifact.of(
            url=task_dir['url'],
            executor=executor.name,
            format=DataFormat.LASP_LAMMPS_OUT_DIR,
            attrs={ **task_dir['attrs'],  'structures_file': 'structures.xyz'},
        ) for task_dir in task_dirs]  # type: ignore
    return CllLaspOutput(output_dirs=output_dirs)


class LammpsInputTemplate(Template):
    delimiter = '$$'


def make_lasp_task_dirs(systems: List[ArtifactDict],
                        lasp_in_data: dict,
                        base_dir: str,
                        type_map: List[str],
                        mass_map: List[float],
                        dp_models: List[str],
                        lammps_input_template: Optional[str],
                        ) -> List[ArtifactDict]:
    input_data = artifacts_to_ase_atoms(systems, type_map=type_map)

    i, task_dirs = 0, []  # TODO: why i is not generated from the loop?
    for artifact, atoms in input_data:
        # create task_dir
        task_dir = os.path.join(base_dir, f'task_{i:06}' )
        os.makedirs(task_dir, exist_ok=True)
        # create lasp.in
        lasp_in_text =  '\n'.join([f'{k:32} {v}' for k, v in lasp_in_data.items()])
        with open(os.path.join(task_dir, 'lasp.in'), 'w', encoding='utf-8') as f:
            f.write(lasp_in_text)
        # create lasp.str
        ase.io.write(os.path.join(task_dir, 'lasp.str'), atoms, format='dmol-arc')
        if lammps_input_template is not None:
            # create lammps.data
            data_file = os.path.join(task_dir, 'lammps.data')
            ase.io.write(data_file, atoms, format='lammps-data', specorder=type_map)  # type: ignore
            # create lammps input: in.simple, PS: Its LASP to blame for the name
            read_data_section = '\n'.join([
                f"read_data {data_file}",
                *(f"mass {i+1} {m}" for i, m in enumerate(mass_map))
            ])
            force_field_section = '\n'.join([
                f"pair_style deepmd {' '.join(dp_models)} out_file {MODEL_DEVI_OUT}",
                f"pair_coeff * *"
            ])
            lammps_input = LammpsInputTemplate(lammps_input_template).substitute(
                read_data_section=read_data_section,
                force_field_section=force_field_section,
            )
            lammps_input_file = os.path.join(task_dir, 'in.simple')
            with open(lammps_input_file, 'w', encoding='utf-8') as f:
                f.write(lammps_input)
        else:
            raise ValueError('At least one potential should be specified.')
        # the `source` field is required as model_devi will use it to update init structures
        task_dirs.append({'url': task_dir,
                            'attrs': {**artifact['attrs'], 'source': artifact['url']}})

        i += 1  # TODO: refactor this
    return task_dirs


def process_lasp_output(task_dir: str, file_name='structures.xyz'):
    """
    Align lasp output with model_devi records.

    As allstr.arc contains all the structures generated by LASP,
    we need to use the result of lasp.out for alignment.

    The following code is copy from ChecMate, may need to be refactored
    """
    all_str_file = os.path.join(task_dir, 'allstr.arc')
    lasp_out_file = os.path.join(task_dir, 'lasp.out')
    all_strs = ase.io.read(all_str_file, ':', format='dmol-arc')

    with open(all_str_file, "r") as f:
        all_qs = list((round(float(line[:73].strip().split()[-2]),6) for line in f.readlines() if "Energy" in line))
    with open(lasp_out_file, "r") as f:
        lines = f.readlines()
        traj_qs = list((round(float(line[:73].strip().split()[2]),6) for line in lines if "Energy,force" in line))
        traj_es = list((round(float(line[:73].strip().split()[1]),6) for line in lines if "Energy,force" in line))

    traj_strs = []
    for i, q in enumerate(all_qs):
        # FIXME: using fuzzy match to align data may have problem in some corner cases
        if len(traj_qs) > 0 and np.isclose(q, traj_qs[0], rtol=0, atol=0.0001):
            all_strs[i].info['ssw_energy'] = traj_es[len(traj_strs)]  # type: ignore
            traj_strs.append(all_strs[i])
            traj_qs.pop(0)
    # write trajectory to file
    ase.io.write(os.path.join(task_dir, file_name), traj_strs, format='extxyz')
    # edit model_devi.out
    model_devi_file = os.path.join(task_dir, MODEL_DEVI_OUT)
    lines = []
    with open(model_devi_file, "r") as f:
        for i, line in enumerate(f):
            if i > 0:
                # replace step 0 with step i so that it can be aligned with structures
                line = re.sub(r'^\s+\d+', f'{i-1:>12} ', line)   #
            lines.append(line)
    with open(model_devi_file, "w") as f:
        f.writelines(lines)

def process_lasp_outputs(task_dirs: List[str], workers: int = 4):
    import joblib
    joblib.Parallel(n_jobs=workers)(
        joblib.delayed(process_lasp_output)(task_dir)
        for task_dir in task_dirs
    )
