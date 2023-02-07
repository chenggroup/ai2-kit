from ai2_kit.core.script import BashTemplate, BashStep, BashScript
from ai2_kit.core.artifact import Artifact
from ai2_kit.core.executor import Executor
from ai2_kit.core.log import get_logger
from ai2_kit.core.job import GatherJobsFuture, retry_fn, map_future

from typing import List, Literal, Optional
from pydantic import BaseModel
from dataclasses import dataclass
from string import Template
import os
import math
import itertools
import random

from .constant import (
    MODEL_DEVI_OUT,
    MODEL_DEVI_NEU_OUT,
    MODEL_DEVI_RED_OUT,
    LAMMPS_TRAJ_DIR,
    LAMMPS_TRAJ_SUFFIX,
)

logger = get_logger(__name__)

class MdVariants(BaseModel):
    system_vars: List[str]  # vars = variants
    temp_vars: List[int]
    pres_vars: List[float]
    lambda_vars: Optional[List[float]]
    trj_freq: int
    nsteps: int
    ensemble: Literal['nvt', 'nvt-i', 'nvt-a', 'nvt-iso', 'nvt-aniso', 'npt', 'npt-t', 'npt-tri', 'nve']


class GeneralLammpsInputConfig(BaseModel):
    iters: List[MdVariants]  # Explore Options for each iteration
    no_pbc: bool = False
    tau_t: float = 0.1
    tau_p: float = 0.5
    timestep: float = 0.0005
    input_template: Optional[str]


class GeneralLammpsContextConfig(BaseModel):
    script_template: BashTemplate
    lammps_cmd: str = 'lmp'
    concurrency: int = 5


@dataclass
class GeneralLammpsInput:

    @dataclass
    class MdOptions:
        models: List[Artifact]

    @dataclass
    class FepOptions:
        red_models: List[Artifact]
        neu_models: List[Artifact]

    config: GeneralLammpsInputConfig
    type_map: List[str]
    mass_map: List[float]

    md_vars: MdVariants
    system_vars: List[Artifact]

    # The following options are mutex
    md_options: Optional[MdOptions] = None
    fep_options: Optional[FepOptions] = None


@dataclass
class GeneralLammpsContext:
    config: GeneralLammpsContextConfig
    path_prefix: str
    executor: Executor

@dataclass
class GeneralLammpsOutput:
    candidates: List[Artifact]

def general_lammps(input: GeneralLammpsInput, ctx: GeneralLammpsContext):
    """
    1. resolve system file path
    2. convert system into lammps format
    3. build configuration file
    4. build script and submit
    """

    # 0. setup dirs
    work_dir = ctx.executor.get_full_path(ctx.path_prefix)
    ctx.executor.mkdir(work_dir)
    logger.info('work_dir is %s', work_dir)

    input_data_dir = os.path.join(work_dir, 'input_data')
    ctx.executor.mkdir(input_data_dir)
    logger.info('data_dir is %s', input_data_dir)

    tasks_dir = os.path.join(work_dir, 'tasks')
    ctx.executor.mkdir(tasks_dir)
    logger.info('tasks_dir is %s', tasks_dir)

    # 1. resolve file path
    system_files_set = set()
    for system_opt in input.system_vars:
        system_files_set.update(ctx.executor.unpack_artifact(system_opt))
    system_files = sorted(system_files_set)

    # 2. convert system into lammps format
    lammps_data_files = []
    zfill_size = math.ceil(math.log10(len(system_files)))
    for i, sys_file in enumerate(system_files):
        file_name = f'{str(i).zfill(zfill_size)}-{os.path.basename(sys_file)}.lammps.data'
        lammps_data_files.append(os.path.join(input_data_dir, file_name))

    def to_lammps_input(input_files: List[str], output_files: List[str], type_map: List[str], fmt='vasp/poscar'):
        import dpdata
        for i in range(len(input_files)):
            dpdata.System(input_files[i], fmt=fmt, type_map=type_map).to_lammps_lmp(output_files[i])  # type: ignore

    ctx.executor.run_python_fn(to_lammps_input)(system_files, lammps_data_files, input.type_map)

    # 3. build configuration files
    md_vars = input.md_vars

    lammps_task_dirs = []
    lammps_input_file_name = 'lammps.input'
    task_counter = itertools.count(0)
    for lammps_data_file in lammps_data_files:
        for temp in md_vars.temp_vars:
            for pres in md_vars.pres_vars:
                task_num = next(task_counter)
                lammps_task_dir = os.path.join(tasks_dir, str(task_num).zfill(zfill_size + 3))  # TODO: optimize zfill
                ctx.executor.mkdir(os.path.join(lammps_task_dir, LAMMPS_TRAJ_DIR))  # create dump directory for lammps or else will get error
                lambda_vars = md_vars.lambda_vars
                lambda_f = lambda_vars[task_num % len(lambda_vars)] if lambda_vars else None

                # TODO: ensure only one of md, fep is set
                if input.md_options:
                    models = [a.url for a in input.md_options.models]
                    force_field_settings = make_md_force_field_settings(models=models)
                    template = input.config.input_template or DEFAULT_MD_INPUT_TEMPLATE

                elif input.fep_options:
                    if lambda_f is None:
                        raise ValueError('lambda_vars must be set when fep is used!')
                    neu_models = [a.url for a in input.fep_options.neu_models]
                    red_models = [a.url for a in input.fep_options.red_models]
                    force_field_settings = make_fep_force_field_settings(neu_models=neu_models, red_models=red_models)
                    template = input.config.input_template or DEFAULT_FEP_INPUT_TEMPLATE
                else:
                    raise ValueError('one and only one of md_options or fep_options must be set')

                input_text = make_lammps_input(
                    template=template,
                    data_file=lammps_data_file,
                    temp=temp,
                    pres=pres,
                    lambda_f=lambda_f,
                    force_field_settings=force_field_settings,

                    nsteps=md_vars.nsteps,
                    trj_freq=md_vars.trj_freq,
                    ensemble=md_vars.ensemble,

                    tau_p=input.config.tau_p,
                    tau_t=input.config.tau_t,
                    timestep=input.config.timestep,

                    mass_map=input.mass_map,
                )
                input_file_path = os.path.join(lammps_task_dir, lammps_input_file_name)
                ctx.executor.dump_text(input_text, input_file_path)
                lammps_task_dirs.append(lammps_task_dir)

    # 4. build scripts and submit
    lammps_cmd = ctx.config.lammps_cmd
    base_cmd = f'{lammps_cmd} -i {lammps_input_file_name}'
    cmd = f'''if [ -f md.restart.10000 ]; then {base_cmd} -v restart 1; else {base_cmd} -v restart 0; fi'''

    # group tasks by concurrency
    concurrency = ctx.config.concurrency
    steps_group = [ list() for _ in range(concurrency)]

    for i, lammps_task_dir in enumerate(lammps_task_dirs):
        steps = steps_group[i % concurrency]
        step = BashStep(
            cwd=lammps_task_dir,
            cmd=cmd,
            checkpoint='lammps',
        )
        steps.append(step)

    # submit jobs
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

    outputs = [
        Artifact(
            url=task_dir,
            executor=ctx.executor.name,
            attrs=dict(by='lammps'),
        ) for task_dir in lammps_task_dirs]  # type: ignore

    future = GatherJobsFuture(jobs, done_fn=retry_fn(max_tries=2), raise_exception=True)
    return map_future(future, GeneralLammpsOutput(
        candidates=outputs
    ))


def make_md_force_field_settings(models: List[str]):
    deepmd_args = ""
    settings = [
        'pair_style deepmd %s out_freq ${THERMO_FREQ} out_file %s %s' % (' '.join(models), MODEL_DEVI_OUT, deepmd_args),
        'pair_coeff * *',
    ]
    return settings


def make_fep_force_field_settings(neu_models: List[str], red_models: List[str]):
    deepmd_args = ""
    settings = [
        'pair_style hybrid/overlay &',
        '           deepmd %s out_freq ${THERMO_FREQ} out_file %s %s &' %(' '.join(neu_models), MODEL_DEVI_NEU_OUT, deepmd_args),
        '           deepmd %s out_freq ${THERMO_FREQ} out_file %s %s'   %(' '.join(red_models), MODEL_DEVI_RED_OUT, deepmd_args),
        'pair_coeff  * * deepmd 1 *',
        'pair_coeff  * * deepmd 2 *',
        '',
        'fix sampling_PES all adapt 0 &',
        '    pair deepmd:1 scale * * v_LAMBDA_f &',
        '    pair deepmd:2 scale * * v_LAMBDA_i &',
        '    scale yes',
    ]
    return settings


def make_lammps_input(template: str,
                      data_file: str,
                      nsteps: int,
                      timestep: float,
                      trj_freq: int,
                      temp: float,
                      pres: float,
                      tau_t: float,
                      tau_p: float,
                      ensemble: str,
                      mass_map: List[float],
                      force_field_settings: List[str],
                      lambda_f: Optional[float] = None,
                      no_pbc=False,
                      rand_start=1_000_000,
                      ):

    # FIXME: Is it a good idea to fix pres or just raise error?
    if not ensemble.startswith('npt'):
        pres = -1

    variables = [
        'variable NSTEPS      equal %d' % nsteps,
        'variable THERMO_FREQ equal %d' % trj_freq,
        'variable DUMP_FREQ   equal %d' % trj_freq,
        'variable TEMP        equal %f' % temp,
        'variable PRES        equal %f' % pres,
        'variable TAU_T       equal %f' % tau_t,
        'variable TAU_P       equal %f' % tau_p,
    ]

    if lambda_f is not None:
        variables.extend([
            '',
            'variable LAMBDA_f    equal %.2f' % lambda_f,
            'variable LAMBDA_i    equal 1-v_LAMBDA_f',
        ])

    init_settings = [
        'boundary ' + ('f f f' if no_pbc else 'p p p'),
    ]

    atom_settings = [
        '''if "${restart} > 0" then "read_restart md.restart.*" else "read_data %s"''' % data_file,
        *("mass {id} {mass}".format(id=i+1, mass=m) for i, m in enumerate(mass_map))
    ]

    md_settings = [
        '''if "${restart} == 0" then "velocity all create ${TEMP} %d"''' % (random.randrange(rand_start - 1) + 1)
    ]

    if ensemble.startswith('npt') and no_pbc:
        raise ValueError('ensemble npt conflict with no_pcb')
    if ensemble in ('npt', 'npt-i', 'npt-iso',):
        md_settings.append('fix 1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}')
    elif ensemble in ('npt-a', 'npt-aniso',):
        md_settings.append('fix 1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}')
    elif ensemble in ('npt-t', 'npt-tri',):
        md_settings.append('fix 1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}')
    elif ensemble in ('nvt',):
        md_settings.append('fix 1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}')
    elif ensemble in ('nve',):
        md_settings.append('fix 1 all nve')
    else:
        raise ValueError('unknown ensemble: ' + ensemble)

    if no_pbc:
        md_settings.extend([
            'velocity all zero linear',
            'fix      fm all momentum 1 linear 1 1 1',
        ])

    md_settings.extend([
        'thermo_style custom step temp pe ke etotal press vol lx ly lz xy xz yz',
        'thermo       ${THERMO_FREQ}',
        'dump         1 all custom ${DUMP_FREQ} %s/*%s id type x y z fx fy fz' % (LAMMPS_TRAJ_DIR, LAMMPS_TRAJ_SUFFIX),
        'restart      10000 md.restart',
    ])

    run_settings = [
        'timestep %f' % timestep,
        'run      ${NSTEPS} upto',
    ]

    return LammpsInputTemplate(template).substitute(dict(
        variable_settings='\n'.join(variables),
        init_settings='\n'.join(init_settings),
        atom_settings='\n'.join(atom_settings),
        md_settings='\n'.join(md_settings),
        force_field_settings='\n'.join(force_field_settings),
        run_settings='\n'.join(run_settings),
    ))


class LammpsInputTemplate(Template):
    """
    change delimiter from $ to $$ as $ is used a lot in lammps input file
    """
    delimiter = '$$'


DEFAULT_MD_INPUT_TEMPLATE = '''# MD INPUT

## VARIABLES
$$variable_settings

## INITIALIZATION SETTINGS
units      metal
atom_style atomic
neighbor   1.0 bin
box        tilt large

$$init_settings

## ATOM SETTINGS
$$atom_settings

change_box all triclinic

## FORCE FIELD SETTINGS
$$force_field_settings

## MD SETTINGS
$$md_settings

## RUN SETTINGS
$$run_settings
'''

DEFAULT_FEP_INPUT_TEMPLATE = '''# FEP INPUT

## VARIABLES
$$variable_settings

variable plus  equal 1
variable minus equal -1

## INITIALIZATION SETTINGS
units        metal
atom_style   atomic
atom_modify  map yes
neighbor     2.0 bin
neigh_modify every 1

$$init_settings

## ATOM SETTINGS
$$atom_settings

change_box all triclinic

## FORCE FIELD SETTINGS
$$force_field_settings

## MD SETTINGS
$$md_settings

## RUN SETTINGS
$$run_settings
'''