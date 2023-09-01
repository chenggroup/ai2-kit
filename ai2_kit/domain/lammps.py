from ai2_kit.core.script import BashTemplate, BashStep, BashScript
from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.log import get_logger
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.util import list_split, dict_nested_get, dump_json, dump_text

from typing import List, Literal, Optional, Mapping, Sequence, Any
from pydantic import BaseModel, validator, root_validator
from dataclasses import dataclass
from string import Template
from allpairspy import AllPairs
from collections import defaultdict

import os
import itertools
import random
import ase.io


from .iface import BaseCllContext, ICllExploreOutput
from .constant import (
    LAMMPS_DUMP_DIR,
    LAMMPS_DUMP_SUFFIX,
    PRESET_LAMMPS_INPUT_TEMPLATE,
)
from .data import DataFormat, artifacts_to_ase_atoms

logger = get_logger(__name__)


class CllLammpsInputConfig(BaseModel):
    n_wise: int = 0
    """
    The way of combining variants.
    if n_wise is less than 2 or greater than total fields, the full combination will be used.
    Or else, the n_wise combination will be used.
    It is strongly recommended to use n_wise when the full combination is too large.
    """

    explore_vars: Mapping[str, List[Any]]
    """
    Variants to be explored.
    Variables defined here will become **LAMMPS variables**.
    If multiple value has been set for a variable,
    the cartesian product will be used to generate the combination.
    For example,

    ```yaml
    TEMP: [330, 430, 530]  # Can be a scalar, e.g. 330
    PRES: 1                # Can be a vector, e.g. [1, 2, 3]
    LAMBDA_f: [0.0, 0.25, 0.5, 0.75, 1.0]
    ```
    Then you can reference them in the LAMMPS input template as ${TEMP}, ${LAMBDA_f}, ${N_STEPS}, etc.
    """
    preset_template: Optional[str] = None
    """
    Name of the preset template.
    """
    input_template: Optional[str] = None
    """
    LAMMPS input template file content.
    If set, the preset_template will be ignored.
    """
    template_vars: Mapping[str, Any] = dict()
    """
    input_template may provide extra injection points for user to inject custom settings.
    Those value could be set here.
    """

    plumed_config: Optional[str]
    """Plumed config file content."""

    system_files: List[str]
    """
    Artifacts key of lammps input data
    """
    ensemble: Literal['nvt', 'nvt-i', 'nvt-a', 'nvt-iso', 'nvt-aniso', 'npt', 'npt-t', 'npt-tri', 'nve', 'csvr']
    no_pbc: bool = False
    nsteps: int
    timestep: float = 0.0005
    sample_freq: int = 100
    mode: Literal['default', 'fep'] = 'default'


    type_alias: Mapping[str, List[str]] = dict()
    '''
    Type alias for atoms. For example, if you want to distinguish ghost H and H of HF molecule from other H atoms,
    you can define the alias as follows:
    ```yaml
    type_alias:
        H: [ H_ghost, H_hf ]
    ```
    And then you can reference them in the LAMMPS input template, for example
    ```
    set atom 1 type ${H_hf}
    set atom 2 type ${H_ghost}
    ```
    '''

    @validator('explore_vars', pre=True)
    @classmethod
    def validate_explore_variants(cls, value):
        if not isinstance(value, dict):
            raise ValueError('explore_vars must be a dict')
        for k in ['TEMP', 'PRES']:
            if k not in value:
                raise ValueError(f'{k} must be set in explore_variants')
        # override default values
        value = {
            'TAU_T': 0.1,
            'TAU_P': 0.5,
            'TIME_CONST': 0.1,
            **value,
        }
        result = {}
        for k, v in value.items():
            if not isinstance(v, list):
                v = [v]
            result[k] = v
        return result

    @root_validator()
    @classmethod
    def validate_domain(cls, values):
        ensemble = values.get('ensemble')
        no_pbc = values.get('no_pbc')
        if ensemble.startswith('npt') and no_pbc:
            raise ValueError('ensemble npt conflict with no_pcb')
        if not ensemble.startswith('npt'):
            logger.info('ensemble is not npt, force PRES to -1')
            values['explore_vars']['PRES'] = [-1]
        return values


class CllLammpsContextConfig(BaseModel):
    script_template: BashTemplate
    lammps_cmd: str = 'lmp'
    concurrency: int = 5
    ignore_error: bool = False


@dataclass
class CllLammpsInput:
    config: CllLammpsInputConfig
    type_map: List[str]
    mass_map: List[float]
    dp_models: Mapping[str, List[Artifact]]
    preset_template: str
    new_system_files: List[Artifact]


@dataclass
class CllLammpsContext(BaseCllContext):
    config: CllLammpsContextConfig


@dataclass
class GenericLammpsOutput(ICllExploreOutput):
    model_devi_outputs: List[Artifact]

    def get_model_devi_dataset(self) -> List[Artifact]:
        return self.model_devi_outputs


async def cll_lammps(input: CllLammpsInput, ctx: CllLammpsContext):
    executor = ctx.resource_manager.default_executor
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)

    if len(input.new_system_files) > 0:
        data_files = input.new_system_files
    else:
        data_files = ctx.resource_manager.resolve_artifacts(input.config.system_files)

    assert len(data_files) > 0, 'no data files found'

    tasks_dir, task_dirs = executor.run_python_fn(make_lammps_task_dirs)(
        combination_vars=input.config.explore_vars,
        data_files=[a.to_dict() for a in data_files],
        dp_models={k: [m.url for m in v] for k, v in input.dp_models.items()},
        n_steps=input.config.nsteps,
        timestep=input.config.timestep,
        sample_freq=input.config.sample_freq,
        no_pbc=input.config.no_pbc,
        n_wise=input.config.n_wise,
        ensemble=input.config.ensemble,
        preset_template=input.config.preset_template or input.preset_template,
        input_template=input.config.input_template,
        plumed_config=input.config.plumed_config,
        extra_template_vars=input.config.template_vars,
        type_map=input.type_map,
        mass_map=input.mass_map,
        type_alias=input.config.type_alias,
        work_dir=work_dir,
    )

    # build scripts and submit
    base_cmd = f'{ctx.config.lammps_cmd} -i lammps.input'
    cmd = f'''if [ -f md.restart.* ]; then {base_cmd} -v restart 1; else {base_cmd} -v restart 0; fi'''

    # generate steps
    steps = []
    for task_dir in task_dirs:
        steps.append(BashStep(
            cwd=task_dir['url'], cmd=cmd, checkpoint='lammps', exit_on_error=not ctx.config.ignore_error))

    # submit jobs by the number of concurrency
    jobs = []
    for i, steps_group in enumerate(list_split(steps, ctx.config.concurrency)):
        if not steps_group:
            continue
        script = BashScript(
            template=ctx.config.script_template,
            steps=steps_group,
        )
        job = executor.submit(script.render(), cwd=tasks_dir,
                              checkpoint_key=f'queue-job/lammps/{i}:{tasks_dir}')
        jobs.append(job)

    await gather_jobs(jobs, max_tries=2)

    # build outputs
    outputs = []
    for task_dir in task_dirs:
        common = dict(url=task_dir['url'], executor=executor.name, format=DataFormat.LAMMPS_OUTPUT_DIR)
        # TODO: a more generic way to dealing multiple model_devi outputs
        if input.config.mode == 'fep':
            outputs += [
                Artifact.of(**common, attrs={
                    **task_dir['attrs'], 'model_devi_file': 'model_devi_ini.out', 'lammps_dump_dir': 'traj-ini',
                    **task_dir['attrs']['fep-ini'],
                }),
                Artifact.of(**common, attrs={
                    **task_dir['attrs'], 'model_devi_file': 'model_devi_fin.out', 'lammps_dump_dir': 'traj-fin',
                    **task_dir['attrs']['fep-fin'],
                    'ancestor': task_dir['attrs']['ancestor'] + '-fin',  # only fin needs
                }),
            ]
        else:
            outputs += [
                Artifact.of(**common, attrs={ **task_dir['attrs'] }),
            ]

    return GenericLammpsOutput(model_devi_outputs=outputs)


def __export_remote_functions():

    class LammpsInputTemplate(Template):
        delimiter = '$$'

    def make_lammps_task_dirs(combination_vars: Mapping[str, Sequence[Any]],
                              data_files: List[ArtifactDict],
                              dp_models: Mapping[str, List[str]],
                              n_steps: int,
                              timestep: float,
                              sample_freq: float,
                              no_pbc: bool,
                              n_wise: int,
                              ensemble: str,
                              preset_template: str,
                              input_template: Optional[str],
                              plumed_config: Optional[str],
                              extra_template_vars: Mapping[str, Any],
                              type_map: List[str],
                              mass_map: List[float],
                              type_alias: Mapping[str, List[str]],
                              work_dir: str,
                              ):
        # setup workspace
        input_data_dir = os.path.join(work_dir, 'input_data')
        tasks_dir = os.path.join(work_dir, 'tasks')
        for path in (input_data_dir, tasks_dir):
            os.makedirs(path, exist_ok=True)

        # create data files
        input_dataset = []
        atoms_list = artifacts_to_ase_atoms(data_files, type_map=type_map)
        for i, (artifact, atoms) in enumerate(atoms_list):
            #  create data file
            data_file = os.path.join(input_data_dir, f'{i:06d}.lammps.data')
            ase.io.write(data_file, atoms, format='lammps-data', specorder=type_map)  # type: ignore
            input_dataset.append({
                'url': data_file,
                'attrs': artifact['attrs'],
            })

        # generate combinations of variants
        combination_fields: List[str] = ['DATA_FILE']
        combination_values: List[List[Any]] = [input_dataset]
        for k, v in combination_vars.items():
            combination_fields.append(k)
            combination_values.append(v)  # type: ignore
        if 1 < n_wise <= len(combination_fields):
            logger.info(f'using {n_wise}-wise combination')
            combinations = AllPairs(combination_values, n=n_wise)
        else:
            logger.info('using full combination')
            combinations = itertools.product(*combination_values)

        # generate tasks input
        task_dirs = []
        for i, combination in enumerate(combinations):
            template_vars = {
                'AI2KIT_CMD': 'ai2-kit',  # TODO: this should be configurable via ctx.config instead of template vars
            }
            lammps_vars = dict(zip(combination_fields, combination))

            # setup task dir
            task_dir = os.path.join(tasks_dir, f'{i:06d}')
            os.makedirs(os.path.join(task_dir, LAMMPS_DUMP_DIR), exist_ok=True)

            data_file = lammps_vars.pop('DATA_FILE')

            # override default values with data file attrs
            overridable_params: dict = dict_nested_get(data_file, ['attrs', 'lammps'], dict())  # type: ignore
            plumed_config = overridable_params.get('plumed_config', plumed_config)
            type_alias = overridable_params.get('type_alias', type_alias)
            extra_template_vars = {**extra_template_vars, **overridable_params.get('template_vars', dict())}

            # build type map and type order
            type_to_mass = dict(zip(type_map, mass_map))

            ext_type_map = []
            ext_type_map_to_origin = []
            ext_mass_map = []
            ghost_loc = []  # location of ghost atom type
            DP_GHOST = len(type_map)

            for origin_type, alias in type_alias.items():
                for t in alias:
                    if 'ghost' in t:  # atom type with 'ghost' in its name is considered as ghost atom type
                        ghost_loc.append(DP_GHOST)

                    ext_type_map.append(t)
                    ext_type_map_to_origin.append(origin_type)
                    ext_mass_map.append(type_to_mass[origin_type])

            # SPECORDER is used to specify the order of types in the lammps data file
            # For example, if the complete type_map is [H, O, O_1, O_2, H_1, H_2],
            # then the specorder should be [H, O, O, O, H, H]
            specorder = type_map + ext_type_map_to_origin

            # *_type_order is use to remap the lammps atom type to dp atom type
            # For example, if the full_type_map is     [H, O, O_1, O_2, H_1, H_2],
            # then the fep_ini_type_order should be    [0, 1, 1  , 1,   0  , 0]
            fep_ini_type_order = [ type_map.index(t) for t in specorder]
            # fep_fin_type_order is the same as fep_ini_type_order, except that the ghost atom type should be len(type_map)
            fep_fin_type_order = fep_ini_type_order.copy()
            for loc in ghost_loc:
                fep_fin_type_order[loc] = DP_GHOST

            template_vars['SPECORDER'] = specorder  # type: ignore
            template_vars['SPECORDER_BASE'] = type_map  # type: ignore

            template_vars['DP_DEFAULT_TYPE_ORDER'] = ' '.join(map(str, range(len(type_map))))
            template_vars['DP_FEP_INI_TYPE_ORDER'] = ' '.join(map(str, fep_ini_type_order))
            template_vars['DP_FEP_FIN_TYPE_ORDER'] = ' '.join(map(str, fep_fin_type_order))

            template_vars['MASS_MAP_FULL'] = _get_masses(type_map + ext_type_map, mass_map + ext_mass_map)
            template_vars['MASS_MAP'] =  template_vars['MASS_MAP_FULL']
            template_vars['MASS_MAP_BASE'] = _get_masses(type_map, mass_map)

            ## build variables section
            lammps_vars['DATA_FILE'] = data_file['url']
            lammps_vars['N_STEPS'] = n_steps
            lammps_vars['THERMO_FREQ'] = sample_freq
            lammps_vars['DUMP_FREQ'] = sample_freq
            lammps_vars['SAMPLE_FREQ'] = sample_freq

            dump_json(lammps_vars, os.path.join(task_dir, 'debug.lammps_vars.json'))  # for debug
            template_vars['VARIABLES'] = _get_lammps_variables(lammps_vars)
            ## build init settings
            template_vars['INITIALIZE'] =  '\n'.join([
                'units           metal',
                'atom_style      atomic',
                'boundary ' + ('f f f' if no_pbc else 'p p p'),
            ])
            ## build read data section
            template_vars['READ_DATA'] = (
                '''if "${restart} > 0" '''
                '''then "read_restart md.restart.*" '''
                '''else "read_data ${DATA_FILE} extra/atom/types %s"''' % (len(ext_type_map))
            )

            ## build simulation
            simulation = [
                '''if "${restart} == 0" then "velocity all create ${TEMP} %d"''' % (random.randrange(10^6 - 1) + 1),
                _get_ensemble(ensemble),
            ]

            if plumed_config:
                plumed_config_file = os.path.join(task_dir, 'plumed.input')
                dump_text(plumed_config, plumed_config_file)
                simulation.append(f'fix cll_plumed all plumed plumedfile {plumed_config_file} outfile plumed.out')

            if no_pbc:
                simulation.extend([
                    'velocity all zero linear',
                    'fix      fm all momentum 1 linear 1 1 1',
                ])
            simulation.extend([
                'thermo_style custom step temp pe ke etotal press vol lx ly lz xy xz yz',
                'thermo       ${THERMO_FREQ}',
                'dump         1 all custom ${DUMP_FREQ} %s/*%s id type x y z fx fy fz' % (LAMMPS_DUMP_DIR, LAMMPS_DUMP_SUFFIX),
                'restart      10000 md.restart',
            ])
            template_vars['SIMULATION'] = '\n'.join(simulation)
            ## build run section
            template_vars['RUN'] = '\n'.join([
                'timestep %f' % timestep,
                'run      ${N_STEPS} upto',
            ])

            dp_models_vars = _get_dp_models_variables(dp_models)

            template_vars = {**template_vars, **dp_models_vars, **extra_template_vars}
            dump_json(template_vars, os.path.join(task_dir, 'debug.template_vars.json'))  # for debug

            input_template = PRESET_LAMMPS_INPUT_TEMPLATE[preset_template] if input_template is None else input_template
            dump_text(input_template, os.path.join(task_dir, 'debug.input_template.txt'))  # for debug
            lammps_input = LammpsInputTemplate(input_template).substitute(defaultdict(str),**template_vars)
            dump_text(lammps_input, os.path.join(task_dir, 'lammps.input'))

            # the `source` field is required as model_devi will use it to update init structures
            task_dirs.append({'url': task_dir,
                              'attrs': {**data_file['attrs'], 'source': data_file['url']}})  # type: ignore
        return tasks_dir, task_dirs


    def _get_type_map_vars(type_map: List[str]):
        return dict(zip(type_map, range(1, len(type_map) + 1)))


    def _get_masses(type_map: List[str], mass_map: List[float]):
        lines = [
            _get_lammps_variables(_get_type_map_vars(type_map)),
            '',
        ]
        for t, m in zip(type_map, mass_map):
            lines.append(f'mass ${{{t}}} {m}')
        return '\n'.join(lines)


    def _get_dp_models_variables(models: Mapping[str, List[str]]):
        vars = {}
        for k, v in models.items():
            prefix = 'DP_MODELS' if k == '' else f'DP_{k}_MODELS'
            vars[prefix] = ' '.join(v)
            for i, m in enumerate(v):
                vars[f'{prefix}_{i}'] = m
        return vars


    def _get_lammps_variables(vars: Mapping[str, Any]):
        lines = []
        for k, v in vars.items():
            if isinstance(v, str):
                # TODO: should escape `v` in case of special characters
                line = f'variable    {k:16} string "{v}"'
            else:
                line = f'variable    {k:16} equal {v}'
            lines.append(line)
        return '\n'.join(lines)


    def _get_ensemble(ensemble: str):
        lines = []
        if ensemble in ('npt', 'npt-i', 'npt-iso',):
            lines.append('fix 1 all npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}')
        elif ensemble in ('npt-a', 'npt-aniso',):
            lines.append('fix 1 all npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}')
        elif ensemble in ('npt-t', 'npt-tri',):
            lines.append('fix 1 all npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}')
        elif ensemble in ('nvt',):
            lines.append('fix 1 all nvt temp ${TEMP} ${TEMP} ${TAU_T}')
        elif ensemble in ('nve',):
            lines.append('fix 1 all nve')
        elif ensemble in ('csvr',):
            lines.append('fix 1 all nve')
            lines.append('fix 2 all temp/csvr ${TEMP} ${TEMP} ${TIME_CONST} %d' % (random.randrange(10^6 - 1) + 1))
        else:
            raise ValueError('unknown ensemble: ' + ensemble)
        return '\n'.join(lines)

    return (
        LammpsInputTemplate,
        make_lammps_task_dirs,
    )


(
    LammpsInputTemplate,
    make_lammps_task_dirs,
) = __export_remote_functions()