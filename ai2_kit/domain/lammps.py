from ai2_kit.core.script import BashTemplate, BashStep, BashScript
from ai2_kit.core.artifact import Artifact
from ai2_kit.core.log import get_logger
from ai2_kit.core.job import GatherJobsFuture, retry_fn
from ai2_kit.core.future import map_future

from typing import List, Literal, Optional, Union, Mapping, Sequence
from pydantic import BaseModel
from dataclasses import dataclass
from string import Template
import os
import itertools
import random


from .cll import BaseCllContext, ICllExploreOutput
from .constant import (
    MODEL_DEVI_OUT,
    MODEL_DEVI_NEU_OUT,
    MODEL_DEVI_RED_OUT,
    LAMMPS_TRAJ_DIR,
    LAMMPS_TRAJ_SUFFIX,
)
from .data_helper import LammpsOutputHelper, PoscarHelper, convert_to_lammps_input_data

logger = get_logger(__name__)

Scalar = Union[str, int, float, bool]

class ExploreVariants(BaseModel):
    temp: List[float]
    """Temperatures variants."""

    pres: List[float]
    """Pressures variants."""

    others: Mapping[str, Sequence[Scalar]] = dict()
    """
    Other variants to be combined with.
    The key is the name of the variant, and the value is the list of values.
    For example, if you want to combine the variant 'LAMBDA' with values [0.0, 0.5, 1.0],
    you can set the others field to {'LAMBDA': [0.0, 0.5, 1.0]}.
    And in LAMMPS input template, you can use the variable ${LAMBDA} and v_LAMBDA to access the value.
    """


class GenericLammpsInputConfig(BaseModel):

    explore_vars: ExploreVariants
    """Variants to be explored."""

    n_wise: int = 0
    """The way of combining variants.
    0 means cartesian product, 2 means 2-wise, etc.
    If n_wise is less than 2 or greater than total fields,
    the full combination will be used.
    It is strongly recommended to use n_wise when the full combination is too large.
    """

    system_files: List[str]
    """Artifacts of initial system data."""

    no_pbc: bool = False
    tau_t: float = 0.1
    tau_p: float = 0.5
    timestep: float = 0.0005
    sample_freq: int
    nsteps: int
    ensemble: Literal['nvt', 'nvt-i', 'nvt-a', 'nvt-iso', 'nvt-aniso', 'npt', 'npt-t', 'npt-tri', 'nve']
    """Ensemble to be used.
    nvt means constant volume and temperature.
    nvt-i means constant volume and temperature, with isotropic scaling.
    nvt-a means constant volume and temperature, with anisotropic scaling.
    nvt-iso means constant volume and temperature, with isotropic scaling.
    npt means constant pressure and temperature.
    npt-t means constant pressure and temperature, with isotropic scaling.
    npt-tri means constant pressure and temperature, with anisotropic scaling.
    nve means constant energy.
    """

    input_template: Optional[str]
    """Lammps input template file content."""

    post_variables_section: str = ''
    post_init_section: str = ''
    post_read_data_section: str = ''
    post_force_field_section: str = ''
    post_md_section: str = ''
    post_run_section: str = ''


class GenericLammpsContextConfig(BaseModel):
    script_template: BashTemplate
    lammps_cmd: str = 'lmp'
    concurrency: int = 5


@dataclass
class GenericLammpsInput:

    @dataclass
    class MdOptions:
        models: List[Artifact]

    @dataclass
    class FepOptions:
        red_models: List[Artifact]
        neu_models: List[Artifact]

    config: GenericLammpsInputConfig
    type_map: List[str]
    mass_map: List[float]

    # The following options are mutex
    md_options: Optional[MdOptions] = None
    fep_options: Optional[FepOptions] = None


@dataclass
class GenericLammpsContext(BaseCllContext):
    config: GenericLammpsContextConfig


@dataclass
class GenericLammpsOutput(ICllExploreOutput):
    model_devi_outputs: List[Artifact]

    def get_model_devi_dataset(self) -> List[Artifact]:
        return self.model_devi_outputs


def generic_lammps(input: GenericLammpsInput, ctx: GenericLammpsContext):
    executor = ctx.resource_manager.default_executor

    # setup workspace
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    input_data_dir, tasks_dir = executor.setup_workspace(work_dir, ['input_data', 'tasks'])

    systems = ctx.resource_manager.resolve_artifacts(input.config.system_files)

    # prepare lammps input data
    # TODO: refactor the way of handling different types of input
    # TODO: handle more data format, for example, cp2k output
    poscar_files: List[Artifact] = []

    for system_file in systems:
        if PoscarHelper.is_match(system_file):
            poscar_files.append(system_file)
        else:
            raise ValueError(f'unsupported system file type: {system_file}')

    input_data_files: List[str] = executor.run_python_fn(convert_to_lammps_input_data)(
        poscar_files=[a.to_dict() for a in poscar_files],
        base_dir=input_data_dir,
        type_map=input.type_map,
    )

    combination_fields: List[str] = [
        'data_file',
        'temp',
        'pres'
    ]
    combination_values: Sequence[Sequence[Scalar]] = [
        input_data_files,
        input.config.explore_vars.temp,
        input.config.explore_vars.pres,
    ]

    for k, v in input.config.explore_vars.others.items():
        combination_fields.append(k)
        combination_values.append(v)  # type: ignore

    if 1 < input.config.n_wise <= len(combination_fields):
        #TODO: implement n-wise combination
        raise NotImplementedError('n-wise combination is not implemented yet')
    else:
        combinations = itertools.product(*combination_values)

    lammps_task_dirs = []
    lammps_input_file_name = 'lammps.input'

    for i, combination in enumerate(combinations):
        data_file, temp, pres = combination[:3]
        others_dict = dict(zip(combination_fields[3:], combination[3:]))
        lammps_task_dir = os.path.join(tasks_dir, str(i).zfill(6))
        executor.mkdir(os.path.join(lammps_task_dir, LAMMPS_TRAJ_DIR))  # create dump directory for lammps or else will get error

        if input.md_options:
            force_field_section = make_md_force_field_section(
                models=[a.url for a in input.md_options.models],
            )
        elif input.fep_options:
            if 'LAMBDA_f' not in others_dict:
                raise ValueError('LAMBDA_f must be set when using FEP mode!')
            # inject the following variables for FEP mode
            others_dict['LAMBDA_i'] = '1-v_LAMBDA_f' # should not have space, or you must quote
            others_dict['plus'] = 1
            others_dict['minus'] = -1
            force_field_section = make_fep_force_field_section(
                neu_models=[a.url for a in input.fep_options.neu_models],
                red_models=[a.url for a in input.fep_options.red_models],
            )
        else:
            raise ValueError('one and only one of md_options or fep_options must be set')
        template = input.config.input_template or  DEFAULT_LAMMPS_INPUT_TEMPLATE

        input_text = make_lammps_input(data_file=data_file,
                          nsteps=input.config.nsteps,
                          timestep=input.config.timestep,
                          trj_freq=input.config.sample_freq,
                          temp=temp,
                          pres=pres,
                          tau_t=input.config.tau_t,
                          tau_p=input.config.tau_p,
                          ensemble=input.config.ensemble,
                          mass_map=input.mass_map,
                          others_dict=others_dict,

                          force_field_section=force_field_section,

                          template=template,
                          post_variables_section=input.config.post_variables_section,
                          post_init_section=input.config.post_init_section,
                          post_read_data_section=input.config.post_read_data_section,
                          post_force_field_section=input.config.post_force_field_section,
                          post_md_section=input.config.post_md_section,
                          post_run_section=input.config.post_run_section,

                          no_pbc=False,
                          rand_start=1_000_000,
                          )

        input_file_path = os.path.join(lammps_task_dir, lammps_input_file_name)
        executor.dump_text(input_text, input_file_path)
        lammps_task_dirs.append(lammps_task_dir)

    # build scripts and submit
    lammps_cmd = ctx.config.lammps_cmd
    base_cmd = f'{lammps_cmd} -i {lammps_input_file_name}'
    cmd = f'''if [ -f md.restart.* ]; then {base_cmd} -v restart 1; else {base_cmd} -v restart 0; fi'''

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
        job = executor.submit(script.render(), cwd=tasks_dir)
        jobs.append(job)

    outputs = [
        Artifact.of(
            url=task_dir,
            executor=executor.name,
            format=LammpsOutputHelper.format,
            attrs=dict(
            ),  # TODO: inherit from input file
        ) for task_dir in lammps_task_dirs]  # type: ignore

    future = GatherJobsFuture(jobs, done_fn=retry_fn(max_tries=2), raise_exception=True)

    return map_future(future, GenericLammpsOutput(
        model_devi_outputs=outputs,
    ))


def make_md_force_field_section(models: List[str]):
    deepmd_args = ""
    settings = [
        'pair_style deepmd %s out_freq ${THERMO_FREQ} out_file %s %s' % (' '.join(models), MODEL_DEVI_OUT, deepmd_args),
        'pair_coeff * *',
    ]
    return settings


def make_fep_force_field_section(neu_models: List[str], red_models: List[str]):
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


def make_lammps_input(data_file: str,
                      nsteps: int,
                      timestep: float,
                      trj_freq: int,
                      temp: float,
                      pres: float,
                      tau_t: float,
                      tau_p: float,
                      ensemble: str,
                      mass_map: List[float],
                      others_dict: Mapping[str, Scalar],

                      template: str,
                      post_variables_section: str,
                      post_init_section: str,
                      post_read_data_section: str,
                      post_force_field_section: str,
                      post_md_section: str,
                      post_run_section: str,

                      force_field_section: List[str],

                      no_pbc=False,
                      rand_start=1_000_000,
                      ):

    # FIXME: I am not sure if it is a good idea to fix it automatically
    # maybe we should consider raise an error here
    if not ensemble.startswith('npt'):
        pres = -1

    variables = [
        '# required variables',
        'variable NSTEPS      equal %d' % nsteps,
        'variable THERMO_FREQ equal %d' % trj_freq,
        'variable DUMP_FREQ   equal %d' % trj_freq,
        'variable TEMP        equal %f' % temp,
        'variable PRES        equal %f' % pres,
        'variable TAU_T       equal %f' % tau_t,
        'variable TAU_P       equal %f' % tau_p,
        '',
        '# custom variables (if any)',
    ]

    for k, v in others_dict.items():
        variables.append(f'variable {k} equal {v}')

    init_section = [
        'boundary ' + ('f f f' if no_pbc else 'p p p'),
    ]

    read_data_section = [
        '''if "${restart} > 0" then "read_restart md.restart.*" else "read_data %s"''' % data_file,
        *("mass {id} {mass}".format(id=i+1, mass=m) for i, m in enumerate(mass_map))
    ]

    md_section = [
        '''if "${restart} == 0" then "velocity all create ${TEMP} %d"''' % (random.randrange(rand_start - 1) + 1)
    ]

    if ensemble.startswith('npt') and no_pbc:
        raise ValueError('ensemble npt conflict with no_pcb')
    if ensemble in ('npt', 'npt-i', 'npt-iso',):
        md_section.append('fix 1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}')
    elif ensemble in ('npt-a', 'npt-aniso',):
        md_section.append('fix 1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}')
    elif ensemble in ('npt-t', 'npt-tri',):
        md_section.append('fix 1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}')
    elif ensemble in ('nvt',):
        md_section.append('fix 1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}')
    elif ensemble in ('nve',):
        md_section.append('fix 1 all nve')
    else:
        raise ValueError('unknown ensemble: ' + ensemble)

    if no_pbc:
        md_section.extend([
            'velocity all zero linear',
            'fix      fm all momentum 1 linear 1 1 1',
        ])

    md_section.extend([
        'thermo_style custom step temp pe ke etotal press vol lx ly lz xy xz yz',
        'thermo       ${THERMO_FREQ}',
        'dump         1 all custom ${DUMP_FREQ} %s/*%s id type x y z fx fy fz' % (LAMMPS_TRAJ_DIR, LAMMPS_TRAJ_SUFFIX),
        'restart      10000 md.restart',
    ])

    run_section = [
        'timestep %f' % timestep,
        'run      ${NSTEPS} upto',
    ]

    return LammpsInputTemplate(template).substitute(dict(
        variables_section='\n'.join(variables),
        post_variables_section=post_variables_section,

        init_section='\n'.join(init_section),
        post_init_section=post_init_section,

        read_data_section='\n'.join(read_data_section),
        post_read_data_section=post_read_data_section,

        md_section='\n'.join(md_section),
        post_md_section=post_md_section,

        force_field_section='\n'.join(force_field_section),
        post_force_field_section=post_force_field_section,

        run_section='\n'.join(run_section),
        post_run_section=post_run_section,
    ))


class LammpsInputTemplate(Template):
    """
    change delimiter from $ to $$ as $ is used a lot in lammps input file
    """
    delimiter = '$$'


DEFAULT_LAMMPS_INPUT_TEMPLATE = '''\
## variables_section
$$variables_section
## post_variables_section
$$post_variables_section

## init_section
units      metal
atom_style atomic
$$init_section
## post_init_section
$$post_init_section

## read_data_section
$$read_data_section
## post_read_data_section
$$post_read_data_section

## force_field_section
$$force_field_section
## post_force_field_section
$$post_force_field_section

## md_section
$$md_section
## post_md_section
$$post_md_section

## run_section
$$run_section
## post_run_section
$$post_run_section
'''
