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
    Variants to be explore by full combination or n_wise combination.

    Variables defined here will become **LAMMPS variables**.
    If multiple value has been set for a variable,
    the cartesian product will be used to generate the combination.
    For example,

    ```yaml
    TEMP: [330, 430, 530]  # Can be a scalar, e.g. 330
    PRES: 1                # Can be a vector, e.g. [1, 2, 3]
    ```
    Then you can reference them in the LAMMPS input template as ${TEMP}, ${LAMBDA_f}, ${N_STEPS}, etc.
    """

    broadcast_vars: Mapping[str, Any] = dict()
    """
    Variants to be explore by broadcast.

    Variables defined here won't join the combination,
    but will be broadcasted to all combinations.

    This can be used to avoid combination explosion.

    ```yaml
    LAMBDA_f: [0.0, 0.25, 0.5, 0.75, 1.0]
    ```
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

    ensemble: Optional[Literal['nvt', 'nvt-i', 'nvt-a', 'nvt-iso', 'nvt-aniso', 'npt', 'npt-t', 'npt-tri', 'nve', 'csvr']] = None
    fix_statement: Optional[str] = None

    no_pbc: bool = False
    nsteps: int
    timestep: float = 0.0005
    sample_freq: int = 100
    mode: Literal['default', 'fep', 'fep-pka', 'fep-redox'] = 'default'

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

    preset_template = input.config.preset_template
    if preset_template is None:
        if input.config.mode in ('fep', 'fep-pka'):
            preset_template = 'fep-pka'
        elif input.config.mode in ('fep-redox',):
            preset_template = 'fep-redox'
        else:
            preset_template = 'default'

    tasks_dir, task_dirs = executor.run_python_fn(make_lammps_task_dirs)(
        combination_vars=input.config.explore_vars,
        broadcast_vars=input.config.broadcast_vars,
        data_files=[a.to_dict() for a in data_files],
        dp_models={k: [m.url for m in v] for k, v in input.dp_models.items()},
        n_steps=input.config.nsteps,
        timestep=input.config.timestep,
        sample_freq=input.config.sample_freq,
        no_pbc=input.config.no_pbc,
        n_wise=input.config.n_wise,
        ensemble=input.config.ensemble,
        fix_statement=input.config.fix_statement,
        preset_template=preset_template,
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
        job = executor.submit(script.render(), cwd=tasks_dir)
        jobs.append(job)

    await gather_jobs(jobs, max_tries=2)

    # build outputs
    outputs = []
    for task_dir in task_dirs:
        common = dict(url=task_dir['url'], executor=executor.name, format=DataFormat.LAMMPS_OUTPUT_DIR)
        if input.config.mode in ('fep', 'fep-pka'):
            # in fep-pka mode,
            # ini and fin states have different structures, so their lammps_dump_dir is different
            # their label method is different too, so we need to unpack `fep-ini` and `fep-fin` accordingly
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
        elif input.config.mode in ('fep-redox',):
            # in fep-redox mode,
            # ini and fin states have the same structure, so just use the default one,
            # but their label method is different, so we need to unpack `fep-ini` and `fep-fin` accordingly
            outputs += [
                Artifact.of(**common, attrs={
                    **task_dir['attrs'], 'model_devi_file': 'model_devi_ini.out',
                    **task_dir['attrs']['fep-ini'],
                }),
                Artifact.of(**common, attrs={
                    **task_dir['attrs'], 'model_devi_file': 'model_devi_fin.out',
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
                              broadcast_vars: Mapping[str, Sequence[Any]],
                              data_files: List[ArtifactDict],
                              dp_models: Mapping[str, List[str]],
                              n_steps: int,
                              timestep: float,
                              sample_freq: float,
                              no_pbc: bool,
                              n_wise: int,
                              ensemble: Optional[str],
                              fix_statement: Optional[str],
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

        combinations = list(map(list, combinations))

        # broadcast broadcast_vars to all combinations
        for k in broadcast_vars.keys():
            # TODO: moving this check to pydatnic validator
            assert k not in combination_fields, f'broadcast_vars {k} is already in explore_vars'

        combination_fields.extend(broadcast_vars.keys())
        for i, combination in enumerate(combinations):
            for _vars in broadcast_vars.values():
                combination.append(_vars[i % len(_vars)])

        # generate tasks input
        task_dirs = []
        for i, combination in enumerate(combinations):
            template_vars = {
                **combination_vars,
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
                    # atom type with 'ghost' in its name is considered as ghost atom type
                    if t.endswith('ghost') or t.endswith('null'):
                        ghost_loc.append(DP_GHOST)
                    ext_type_map.append(t)
                    ext_type_map_to_origin.append(origin_type)
                    ext_mass_map.append(type_to_mass[origin_type])

            # SPECORDER is used to specify the order of types in the lammps data file
            # For example, if the complete type_map is [H, O, O_1, O_2, H_1, H_2],
            # then the specorder should be [H, O, O, O, H, H]
            specorder = type_map + ext_type_map_to_origin

            # Since deepmd 2.2.4 its lammps module support specify type map via pair coeff,
            # So we don't need to use the type order hack any more.
            # ref: https://github.com/deepmodeling/deepmd-kit/pull/2732
            fep_fin_specorder = specorder.copy()
            for loc in ghost_loc:
                fep_fin_specorder[loc] = 'NULL'

            # specorder in the format of H O H NULL, for lammps pair coeff input
            template_vars['SPECORDER'] = ' '.join(specorder)
            template_vars['SPECORDER_BASE'] = ' '.join(type_map)
            template_vars['FEP_INI_SPECORDER'] = template_vars['SPECORDER']
            template_vars['FEP_FIN_SPECORDER'] = ' '.join(fep_fin_specorder)

            # specorder in the format of ['H', 'O', 'H', 'NULL'], for ai2-iit command line input
            template_vars['SPECORDER_LIST'] = str(specorder)
            template_vars['SPECORDER_BASE_LIST'] = str(type_map)
            template_vars['FEP_INI_SPECORDER_LIST'] = template_vars['SPECORDER_LIST']
            template_vars['FEP_FIN_SPECORDER_LIST'] = str(fep_fin_specorder)

            # mass map in the form of
            # mass ${H} 1.007
            # mass ${O} 15.999
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
            if fix_statement is None:
                assert ensemble is not None, 'either fix_statement or ensemble is required'
                fix_statement = get_ensemble(ensemble)

            simulation = [
                '''if "${restart} == 0" then "velocity all create ${TEMP} %d"''' % (random.randrange(10^6 - 1) + 1),
                fix_statement,
            ]

            if plumed_config:
                plumed_config = LammpsInputTemplate(plumed_config).substitute(defaultdict(str), **template_vars)
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

            if input_template is None:
                input_template = PRESET_LAMMPS_INPUT_TEMPLATE[preset_template]
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


    def get_ensemble(ensemble: str):
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
        get_ensemble,
    )


(
    LammpsInputTemplate,
    make_lammps_task_dirs,
    get_ensemble,
) = __export_remote_functions()