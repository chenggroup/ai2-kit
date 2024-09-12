from ai2_kit.core.script import BashTemplate, BashStep, BashScript, make_gpu_parallel_steps
from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.log import get_logger
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.util import list_split, dict_nested_get, dump_json, dump_text
from ai2_kit.core.pydantic import BaseModel

from typing import List, Literal, Optional, Mapping, Sequence, Any
from pydantic import field_validator, model_validator
from dataclasses import dataclass
from string import Template
from allpairspy import AllPairs
from collections import defaultdict

import os
import itertools
import random
import ase.io

from .iface import BaseCllContext, ICllExploreOutput, TRAINING_MODE
from .constant import (
    PRESET_LAMMPS_INPUT_TEMPLATE,
)
from .data import DataFormat, artifacts_to_ase_atoms
from .dpff import dump_dplr_lammps_data

logger = get_logger(__name__)


class FepOptions(BaseModel):
    ini_ghost_types: List[str] = []
    fin_ghost_types: List[str] = []


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

    Those vars can be referenced in the LAMMPS input template as $$VAR_NAME.
    """

    plumed_config: Optional[str] = None
    """Plumed config file content."""

    system_files: List[str]
    """
    Artifacts key of lammps input data
    """

    ensemble: Optional[Literal['nvt', 'npt', 'npt-i', 'npt-a', 'npt-iso', 'npt-aniso', 'npt-t', 'npt-tri', 'npt-x', 'npt-y', 'npt-z', 'nve', 'csvr']] = None
    fix_statement: Optional[str] = None

    no_pbc: bool = False
    nsteps: int
    timestep: float = 0.0005
    sample_freq: int = 100
    ignore_error: bool = False

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

    custom_ff: Optional[str] = None
    '''
    Allow user to set custom force field. If set, the custom force field will be used instead of the default one.
    The use can use $$DP_MODELS to reference the deepmd models, and $$SPECORDER to reference the atom type order.
    For example:

    pair_style hybrid/overlay &
               deepmd $$DP_MODELS out_freq ${THERMO_FREQ} out_file model_devi.out $$FEP_DP_OPT &
               buck/coul/long 10.0 10.0

    pair_coeff  * * deepmd 1 $$SPECORDER
    pair_coeff  * * buck/coul/long 10.0 10.0
    '''

    fep_opts: FepOptions = FepOptions()


    @field_validator('explore_vars', mode='before')
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


    @model_validator(mode='before')
    @classmethod
    def validate_domain(cls, values):
        ensemble = values.get('ensemble')
        if ensemble:
            no_pbc = values.get('no_pbc')
            if ensemble.startswith('npt') and no_pbc:
                raise ValueError('ensemble npt conflict with no_pcb')
            if not ensemble.startswith('npt'):
                logger.info('ensemble is not npt, force PRES to -1')
                values['explore_vars']['PRES'] = [-1]

        # get all alias types
        type_alias = values.get('type_alias', {})
        alias_type = list(itertools.chain(*type_alias.values()))

        # ensure all ghost type are defined in type_alias
        fep_opts = values.get('fep_opts')
        if fep_opts:
            ghost_types = fep_opts.ini_ghost_types + fep_opts.fin_ghost_types
            for t in ghost_types:
                if t not in alias_type:
                    raise ValueError(f'ghost type {t} is not defined in type_alias')

        return values

    def assert_var(self, var: str, msg: str = ''):
        if not msg:
            msg = f'{var} is not defined in explore_vars or broadcast_vars'
        assert var in self.explore_vars or var in self.broadcast_vars, msg


class CllLammpsContextConfig(BaseModel):
    script_template: BashTemplate
    lammps_cmd: str = 'lmp'
    concurrency: int = 5
    multi_gpus_per_job: bool = False

@dataclass
class CllLammpsInput:
    config: CllLammpsInputConfig
    type_map: List[str]
    mass_map: List[float]
    mode: TRAINING_MODE
    preset_template: str
    new_system_files: List[Artifact]
    dp_models: Mapping[str, List[Artifact]]
    dp_modifier: Optional[dict]
    dp_sel_type: Optional[List[int]]


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
        if input.mode == 'fep-pka':
            preset_template = 'fep-pka'
        elif input.mode == 'fep-redox':
            preset_template = 'fep-redox'
        elif input.mode == 'dpff':
            preset_template = 'dpff'
        else:
            preset_template = 'custom-ff' if input.config.custom_ff else 'default'

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
        mode=input.mode,
        dp_modifier=input.dp_modifier,
        dp_sel_type=input.dp_sel_type,
        fep_opts=input.config.fep_opts,
        custom_ff=input.config.custom_ff,
    )

    # build scripts and submit
    base_cmd = f'{ctx.config.lammps_cmd} -i lammps.input'
    cmd = f'''if [ -f md.restart.* ]; then {base_cmd} -v restart 1; else {base_cmd} -v restart 0; fi'''

    # generate steps
    steps = []
    for task_dir in task_dirs:
        steps.append(BashStep(
            cwd=task_dir['url'], cmd=cmd, checkpoint='lammps', exit_on_error=not input.config.ignore_error))

    # submit jobs by the number of concurrency
    jobs = []
    for i, steps_group in enumerate(list_split(steps, ctx.config.concurrency)):
        if not steps_group:
            continue
        if ctx.config.multi_gpus_per_job:
            script = BashScript(
                template=ctx.config.script_template,
                steps=make_gpu_parallel_steps(steps_group),
            )
        else:
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
        if input.mode == 'fep-pka':
            # in fep-pka mode,
            # ini and fin states have different structures, so their structure files are different
            # their label method is different too, so we need to unpack `fep-ini` and `fep-fin` accordingly
            outputs += [
                Artifact.of(**common, attrs={
                    **task_dir['attrs'],
                    'model_devi_file': 'model_devi_ini.out',
                    'structures': 'traj-fep-ini.lammpstrj',
                    **task_dir['attrs']['fep-ini'],
                }),
                Artifact.of(**common, attrs={
                    **task_dir['attrs'],
                    'model_devi_file': 'model_devi_fin.out',
                    'structures': 'traj-fep-fin.lammpstrj',
                    **task_dir['attrs']['fep-fin'],
                    'ancestor': task_dir['attrs']['ancestor'] + '-fin',  # only fin needs
                }),
            ]
        elif input.mode == 'fep-redox':
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
                          dp_modifier: Optional[dict],
                          dp_sel_type: Optional[List[int]],
                          mode: TRAINING_MODE,
                          fep_opts: FepOptions,
                          rel_path: bool = False,
                          custom_ff: Optional[str] = None,
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
        ancestor = artifact['attrs']['ancestor']
        data_file = os.path.join(input_data_dir, f'{ancestor}-{i:06d}.lammps.data')
        if mode == 'dpff':
            assert dp_modifier is not None and dp_sel_type is not None, 'dp_modifier & dp_sel_type is required for dpff mode'
            sys_charge_map = dp_modifier['sys_charge_map']
            model_charge_map = dp_modifier['model_charge_map']
            with open(data_file, 'w') as fp:
                dump_dplr_lammps_data(fp, atoms, type_map=type_map, sel_type=dp_sel_type,
                                      sys_charge_map=sys_charge_map, model_charge_map=model_charge_map)  # type: ignore
        else:
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
        lammps_vars = dict(zip(combination_fields, combination))
        template_vars = {
            **lammps_vars,
            'CUSTOM_FF': custom_ff or '',
        }

        # setup task dir
        task_dir = os.path.join(tasks_dir, f'{i:06d}')
        os.makedirs(task_dir, exist_ok=True)

        data_file = lammps_vars.pop('DATA_FILE')

        # override default values with data file attrs
        overridable_params: dict = dict_nested_get(data_file, ['attrs', 'lammps'], dict())  # type: ignore
        _plumed_config = overridable_params.get('plumed_config', plumed_config)
        _fix_statement = overridable_params.get('fix_statement', fix_statement)
        _ensemble = overridable_params.get('ensemble', ensemble)
        _type_alias = overridable_params.get('type_alias', type_alias)
        _fep_opts = FepOptions(**overridable_params.get('fep_opts', fep_opts.dict()))

        # be careful to override template_vars without changing the original dict
        _extra_template_vars = {**extra_template_vars, **overridable_params.get('template_vars', dict())}

        # generate types related template vars
        types_template_vars = get_types_template_vars(
            type_map=type_map, mass_map=mass_map,
            type_alias=_type_alias, sel_type=dp_sel_type,
            fep_ini_ghost_types=_fep_opts.ini_ghost_types,
            fep_fin_ghost_types=_fep_opts.fin_ghost_types,
        )

        ## build variables section
        if rel_path:
            lammps_vars['DATA_FILE'] = os.path.relpath(data_file['url'], task_dir)
        else:
            lammps_vars['DATA_FILE'] = data_file['url']

        lammps_vars['N_STEPS'] = n_steps
        lammps_vars['THERMO_FREQ'] = sample_freq
        lammps_vars['DUMP_FREQ'] = sample_freq
        lammps_vars['SAMPLE_FREQ'] = sample_freq
        lammps_vars['DEFAULT_GROUP'] = 'all'

        if mode == 'dpff':
            assert dp_modifier is not None
            lammps_vars['DEFAULT_GROUP'] = 'real_atom'
            lammps_vars['EWALD_BETA'] = dp_modifier['ewald_beta']
        lammps_vars['DUMP_GROUP'] = lammps_vars['DEFAULT_GROUP']

        dump_json(lammps_vars, os.path.join(task_dir, 'debug.lammps_vars.json'))  # for debug
        template_vars['VARIABLES'] = _get_lammps_variables(lammps_vars)
        ## build init settings
        template_vars['INITIALIZE'] =  '\n'.join([
            'units           metal',
            'atom_style      %s' % ('full' if mode == 'dpff' else 'atomic'),
            'boundary ' + ('f f f' if no_pbc else 'p p p'),
        ])
        ## build read data section
        extra_types = sum(len(l) for l in _type_alias.values())  # how many alias type are defined
        template_vars['READ_DATA'] = (
            '''if "${restart} > 0" '''
            '''then "read_restart md.restart.*" '''
            '''else "read_data ${DATA_FILE} extra/atom/types %s"''' % extra_types
        )

        ## build simulation section
        simulation = [
            '''if "${restart} == 0" then "velocity ${DEFAULT_GROUP} create ${TEMP} %d"''' % (random.randrange(10 ^ 6 - 1) + 1)
        ]

        if _fix_statement is None:
            assert _ensemble is not None, 'either fix_statement or ensemble is required'
            _fix_statement = get_ensemble(_ensemble, group='${DEFAULT_GROUP}')

        if mode == 'dpff':
            simulation.extend([
                'compute  real_temp real_atom temp',
                _fix_statement,
                'fix_modify 1 temp real_temp',
                '',
            ])
        else:
            simulation.append(_fix_statement)

        if _plumed_config:
            _plumed_config = LammpsInputTemplate(_plumed_config).substitute(defaultdict(str), **template_vars)
            plumed_config_file = os.path.join(task_dir, 'plumed.input')
            dump_text(_plumed_config, plumed_config_file)
            simulation.append(f'fix cll_plumed ${{DEFAULT_GROUP}} plumed plumedfile plumed.input outfile plumed.out')

        if no_pbc:
            simulation.extend([
                'velocity ${DEFAULT_GROUP} zero linear',
                'fix      fm ${DEFAULT_GROUP} momentum 1 linear 1 1 1',
            ])
        simulation.extend([
            'thermo_style custom step temp pe ke etotal press vol lx ly lz xy xz yz',
            'thermo       ${THERMO_FREQ}',
        ])

        if mode == 'fep-pka':
            simulation.extend([
                'dump 1 fep_ini_atoms custom ${DUMP_FREQ} traj-fep-ini.lammpstrj id type element x y z fx fy fz',
                'dump 2 fep_fin_atoms custom ${DUMP_FREQ} traj-fep-fin.lammpstrj id type element x y z fx fy fz',
                f'dump_modify 1 element {types_template_vars["SPECORDER"]}',
                f'dump_modify 2 element {types_template_vars["SPECORDER"]}',
            ])
        else:
            simulation.extend([
                'dump 1 ${DUMP_GROUP} custom ${DUMP_FREQ} traj.lammpstrj id type element x y z fx fy fz',
                f'dump_modify 1 element {types_template_vars["SPECORDER"]}',
            ])
        simulation.append('restart 10000 md.restart')

        template_vars['SIMULATION'] = '\n'.join(simulation)
        ## build run section
        template_vars['RUN'] = '\n'.join([
            'timestep %f' % timestep,
            'run      ${N_STEPS} upto',
        ])

        dp_models_vars = _get_dp_models_variables(dp_models)
        template_vars = {**template_vars, **types_template_vars, **dp_models_vars, **_extra_template_vars}
        dump_json(template_vars, os.path.join(task_dir, 'debug.template_vars.json'))

        if input_template is None:
            input_template = PRESET_LAMMPS_INPUT_TEMPLATE[preset_template]
        dump_text(input_template, os.path.join(task_dir, 'debug.input_template.txt'))
        lammps_input = LammpsInputTemplate(input_template).substitute(defaultdict(str),**template_vars)
        dump_text(lammps_input, os.path.join(task_dir, 'lammps.input'))

        # the `source` field is required as model_devi will use it to update init structures
        task_dirs.append({'url': task_dir,
                          'attrs': {
                              **data_file['attrs'],
                              'source': data_file['url'],
                              'efield': lammps_vars.get('EFIELD'),
                          }})  # type: ignore
    return tasks_dir, task_dirs


def get_types_template_vars(type_map: List[str], mass_map: List[float],
                            type_alias: Mapping[str, List[str]], sel_type: Optional[List[int]],
                            fep_ini_ghost_types: List[str], fep_fin_ghost_types: List[str]):
    """
    generate template vars that related to type_map, mass_map, type_alias, sel_type

    the order of atom type index is:
    real atoms (defined in type_map), virtual atoms (defined in sel_type) and then alias (defined in type_alias)
    """
    template_vars = {}
    type_to_mass = dict(zip(type_map, mass_map))

    # new types gonna to be added
    ext_type_map = []
    ext_mass_map = []

    # handle sel_type (used by dplr)
    # sel_type must be handle before type_alias as they are defined in data file
    # while type_alias are defined in lammps script
    type_association = []
    if sel_type is not None:
        n_real_atom = len(type_map)
        for i, t in enumerate(sel_type):
            # add placeholder for sel_type
            ext_type_map.append(f'_X_{i}')
            ext_mass_map.append(1.0)
            # type association is to define relationship between virtual and real atom type in lammps
            type_association.extend([t + 1, n_real_atom + i + 1])
        template_vars['DPLR_TYPE_ASSOCIATION'] = ' '.join(map(str, type_association))

    # SPECORDER is used to specify the order of types in the lammps data file
    # For example, if the complete type_map is [H, O, O_1, O_2, H_1, H_2],
    # then the specorder should be [H, O, O, O, H, H]
    specorder = type_map[:]

    fep_ini_specorder = type_map[:]
    fep_fin_specorder = type_map[:]

    for real_type, alias in type_alias.items():
        for t in alias:
            specorder.append(real_type)
            if t in fep_ini_ghost_types:
                fep_ini_specorder.append('NULL')
            else:
                fep_ini_specorder.append(real_type)
            if t in fep_fin_ghost_types:
                fep_fin_specorder.append('NULL')
            else:
                fep_fin_specorder.append(real_type)

            ext_type_map.append(t)
            ext_mass_map.append(type_to_mass[real_type])

    # define group for fep pka mode
    all_types = type_map + ext_type_map

    fep_ini_type_vars = _to_lammps_type_vars([t for t in all_types if t not in fep_ini_ghost_types])
    fep_fin_type_vars = _to_lammps_type_vars([t for t in all_types if t not in fep_fin_ghost_types])

    template_vars['FEP_GROUPS'] = '\n'.join([
        f'group fep_ini_atoms type {fep_ini_type_vars}',
        f'group fep_fin_atoms type {fep_fin_type_vars}'
    ])

    # define group for dpff mode
    if sel_type is not None:
        real_atom_start = 1
        real_atom_end = real_atom_start + len(type_map)
        virtual_atom_end = real_atom_end + len(sel_type)
        alias_end = virtual_atom_end + len(ext_type_map) - len(sel_type)

        dpff_real_atom = [*range(real_atom_start, real_atom_end),  *range(virtual_atom_end, alias_end)]
        dpff_virtual_atom = range(real_atom_end, virtual_atom_end)

        template_vars['DPFF_REAL_ATOM'] = ' '.join(map(str, dpff_real_atom))
        template_vars['DPFF_VIRTUAL_ATOM'] = ' '.join(map(str, dpff_virtual_atom))
        template_vars['DPFF_GROUPS'] = '\n'.join([
            f'group real_atom    type {template_vars["DPFF_REAL_ATOM"]}',
            f'group virtual_atom type {template_vars["DPFF_VIRTUAL_ATOM"]}',
            f'neigh_modify    every 10 delay 0 check no exclude group real_atom virtual_atom',
        ])

    # specorder in the format of H O H NULL, for lammps pair coeff input
    template_vars['SPECORDER'] = ' '.join(specorder)

    template_vars['FEP_INI_SPECORDER'] = ' '.join(fep_ini_specorder)
    template_vars['FEP_FIN_SPECORDER'] = ' '.join(fep_fin_specorder)

    # mass map is in the form of
    # variable   H               equal 1
    # variable   O               equal 2
    # variable   H_null          equal 3
    # mass ${H} 1.007
    # mass ${O} 15.999
    # mass ${H_null} 1.0
    template_vars['MASS_MAP'] = _get_masses(type_map + ext_type_map, mass_map + ext_mass_map)
    return template_vars


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
        elif isinstance(v, list):  # vector or args
            line = f'variable    {k:16} string \"{" ".join(str(x) for x in v)}\"'
        else:
            line = f'variable    {k:16} equal {v}'
        lines.append(line)
    return '\n'.join(lines)


def get_ensemble(ensemble: str, group='all'):
    lines = []
    if ensemble in ('npt', 'npt-i', 'npt-iso',):
        lines.append('fix 1 %(group)s npt temp ${TEMP} ${TEMP} ${TAU_T} iso ${PRES} ${PRES} ${TAU_P}')
    elif ensemble in ('npt-a', 'npt-aniso',):
        lines.append('fix 1 %(group)s npt temp ${TEMP} ${TEMP} ${TAU_T} aniso ${PRES} ${PRES} ${TAU_P}')
    elif ensemble in ('npt-t', 'npt-tri',):
        lines.append('fix 1 %(group)s npt temp ${TEMP} ${TEMP} ${TAU_T} tri ${PRES} ${PRES} ${TAU_P}')
    elif ensemble in ('npt-x',):
        lines.append('fix 1 %(group)s npt temp ${TEMP} ${TEMP} ${TAU_T} x ${PRES} ${PRES} ${TAU_P} y 0 0 ${TAU_P} z 0 0 ${TAU_P}')
    elif ensemble in ('npt-y',):
        lines.append('fix 1 %(group)s npt temp ${TEMP} ${TEMP} ${TAU_T} y ${PRES} ${PRES} ${TAU_P} x 0 0 ${TAU_P} z 0 0 ${TAU_P}')
    elif ensemble in ('npt-z',):
        lines.append('fix 1 %(group)s npt temp ${TEMP} ${TEMP} ${TAU_T} z ${PRES} ${PRES} ${TAU_P} x 0 0 ${TAU_P} y 0 0 ${TAU_P}')
    elif ensemble in ('nvt',):
        lines.append('fix 1 %(group)s nvt temp ${TEMP} ${TEMP} ${TAU_T}')
    elif ensemble in ('nve',):
        lines.append('fix 1 %(group)s nve')
    elif ensemble in ('csvr',):
        lines.append('fix 1 %(group)s nve')
        lines.append('fix 2 %(group)s temp/csvr ${TEMP} ${TEMP} ${TIME_CONST} %(seed)d')
    else:
        raise ValueError('unknown ensemble: ' + ensemble)
    return '\n'.join(lines) % {'group': group, 'seed': random.randrange(10^6 - 1) + 1}


def _to_lammps_type_vars(types: List[str]):
    return ' '.join(f'${{{t}}}' for t in types)
