from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.script import BashScript, BashStep, BashTemplate
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.util import dict_nested_get, dump_json, list_split, list_sample
from ai2_kit.core.log import get_logger
from ai2_kit.core.pydantic import BaseModel

from typing import List, Union, Optional, Tuple, Literal
from pymatgen.io.vasp.inputs import Incar, Kpoints
from dataclasses import dataclass

import ase.io
from ase import Atoms
from ase.io.vasp import _symbol_count_from_symbols

import copy
import os

from .data import DataFormat, artifacts_to_ase_atoms
from .iface import ICllLabelOutput, BaseCllContext

logger = get_logger(__name__)

class CllVaspInputConfig(BaseModel):
    init_system_files: List[str] = []
    input_template: Optional[Union[dict, str]] = None
    """
    INCAR template for VASP. Could be a dict or content of a VASP input file.
    """
    potcar_source: Optional[dict] = {}
    """
    POTCAR source for VASP. Could be a dict or list of paths.
    """
    kpoints_template: Optional[Union[dict, str]] = None
    """
    KPOINTS template for VASP. Could be a dict or content of a VASP input file.
    """
    limit: int = 50
    """
    Limit of the number of systems to be labeled.
    """
    limit_method: Literal["even", "random", "truncate"] = "even"

    ignore_error: bool = False
    """
    Ignore error when running VASP.
    """

class CllVaspContextConfig(BaseModel):
    script_template: BashTemplate
    vasp_cmd: str = 'vasp_std'
    post_vasp_cmd: str = 'echo "no post_vasp_cmd"'
    concurrency: int = 5

@dataclass
class CllVaspInput:
    config: CllVaspInputConfig
    system_files: List[Artifact]
    type_map: List[str]
    initiated: bool = False  # FIXME: this seems to be a bad design idea


@dataclass
class CllVaspContext(BaseCllContext):
    config: CllVaspContextConfig


@dataclass
class GenericVaspOutput(ICllLabelOutput):
    vasp_outputs: List[Artifact]

    def get_labeled_system_dataset(self):
        return self.vasp_outputs


async def cll_vasp(input: CllVaspInput, ctx: CllVaspContext) -> GenericVaspOutput:
    executor = ctx.resource_manager.default_executor

    # For the first round
    # FIXME: move out from this function, this should be done in the workflow
    if not input.initiated:
        input.system_files += ctx.resource_manager.resolve_artifacts(input.config.init_system_files)

    if len(input.system_files) == 0:
        return GenericVaspOutput(vasp_outputs=[])

    # setup workspace
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    [tasks_dir] = executor.setup_workspace(work_dir, ['tasks'])

    # create task dirs and prepare input files
    vasp_task_dirs = executor.run_python_fn(make_vasp_task_dirs)(
        system_files=[a.to_dict() for a in input.system_files],
        type_map=input.type_map,
        base_dir=tasks_dir,
        input_template=input.config.input_template,
        potcar_source=input.config.potcar_source,
        kpoints_template=input.config.kpoints_template,
        limit=input.config.limit,
        limit_method=input.config.limit_method,
    )

    # build commands
    steps = []
    for vasp_task_dir in vasp_task_dirs:
        cmd = f'{ctx.config.vasp_cmd} &> output && {ctx.config.post_vasp_cmd}'
        steps.append(BashStep(
            cwd=vasp_task_dir['url'],
            cmd=cmd,
            checkpoint='vasp',
            exit_on_error=not input.config.ignore_error,
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
        job = executor.submit(script.render(), cwd=tasks_dir)
        jobs.append(job)
    jobs = await gather_jobs(jobs, max_tries=2)

    vasp_outputs = [Artifact.of(
        url=a['url'],
        format=DataFormat.VASP_OUTPUT_DIR,
        executor=executor.name,
        attrs=a['attrs'],
    ) for a in vasp_task_dirs]

    return GenericVaspOutput(vasp_outputs=vasp_outputs)


def make_vasp_task_dirs(system_files: List[ArtifactDict],
                        type_map: List[str],
                        input_template: Optional[Union[dict, str]],
                        potcar_source: Optional[dict],
                        base_dir: str,
                        kpoints_template: Optional[Union[dict, str]] = None,
                        limit: int = 0,
                        limit_method: Literal["even", "random", "truncate"] = "even"
                        ) -> List[ArtifactDict]:
    """Generate VASP input files from LAMMPS dump files or XYZ files."""

    task_dirs = []
    atoms_list: List[Tuple[ArtifactDict, Atoms]] = artifacts_to_ase_atoms(system_files, type_map=type_map)

    if limit > 0:
        atoms_list = list_sample(atoms_list, limit, method=limit_method)

    for i, (data_file, atoms) in enumerate(atoms_list):
        # create task dir
        task_dir = os.path.join(base_dir, f'{str(i).zfill(6)}')
        os.makedirs(task_dir, exist_ok=True)
        dump_json(data_file, os.path.join(task_dir, 'debug.data-file.json'))

        # load system-wise config from attrs
        overridable_params: dict = copy.deepcopy(dict_nested_get(data_file, ['attrs', 'vasp'], dict()))  # type: ignore

        # create input file
        input_template = overridable_params.get('input_template', input_template)

        # prepare input template
        if isinstance(input_template, str):
            input_template = Incar.from_file(input_template).as_dict()

        assert input_template, 'input_template must be provided'
        incar = Incar.from_dict(input_template)
        incar.write_file(os.path.join(task_dir, 'INCAR'))

        # create POSCAR
        elements_all = atoms.get_chemical_symbols()
        elements = [
            item[0] for item in _symbol_count_from_symbols(elements_all)
        ]
        ase.io.write(
            os.path.join(task_dir, 'POSCAR'), atoms, format='vasp'
        )

        # create POTCAR
        potcar_source = overridable_params.get('potcar_source', potcar_source)

        # prepare potcar
        assert potcar_source, 'potcar_source must be provided'
        with open(os.path.join(task_dir, 'POTCAR'), 'w') as out_file:
            for element in elements:
                with open(potcar_source[element], 'r') as in_file:
                    out_file.write(in_file.read())

        # create KPOINTS
        kpoints_template = overridable_params.get('kpoints_template', kpoints_template)

        # prepare kpoints template
        if isinstance(kpoints_template, str):
            kpoints_template = Kpoints.from_file(kpoints_template).as_dict()
        
        if kpoints_template:
            kpoints = Kpoints.from_dict(kpoints_template)
            kpoints.write_file(os.path.join(task_dir, 'KPOINTS'))

        # inherit attrs from input file
        # TODO: inherit only ancestor key should be enough
        task_dirs.append({
            'url': task_dir,
            'attrs': data_file['attrs'],
        })

    return task_dirs
