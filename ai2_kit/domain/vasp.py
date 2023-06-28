from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.script import BashScript, BashStep, BashTemplate
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.util import merge_dict, dict_nested_get, dict_nested_set, split_list
from ai2_kit.core.log import get_logger

from typing import List, Union
from pydantic import BaseModel
from dataclasses import dataclass

from typing import List, Union, Optional, Tuple
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pydantic import BaseModel
from dataclasses import dataclass

import copy
import os

from .data_helper import LammpsOutputHelper, XyzHelper, VaspOutputHelper
from .cll import ICllLabelOutput, BaseCllContext

logger = get_logger(__name__)

class GenericVaspInputConfig(BaseModel):
    init_system_files: List[str] = []
    limit: int = 50
    input_template: Union[dict, str]
    potcar_source: Union[dict, list]
    kpoints_template: Optional[str] = None
    """
    Input template for VASP. Could be a dict or content of a VASP input file.

    Note:
    If you are using files in input templates, it is recommended to use artifact name instead of literal path.
    String starts with '@' will be treated as artifact name.
    For examples, input_template = @vasp/INCAR.
    You can still use literal path, but it is not recommended.
    """

class GenericVaspContextConfig(BaseModel):
    script_template: BashTemplate
    vasp_cmd: str = 'vasp_std'
    concurrency: int = 5

@dataclass
class GenericVaspInput:
    config: GenericVaspInputConfig
    system_files: List[Artifact]
    type_map: List[str]
    initiated: bool = False


@dataclass
class GenericVaspContext(BaseCllContext):
    config: GenericVaspContextConfig


@dataclass
class GenericVaspOutput(ICllLabelOutput):
    vasp_outputs: List[Artifact]

    def get_labeled_system_dataset(self):
        return self.vasp_outputs


async def generic_vasp(input: GenericVaspInput, ctx: GenericVaspContext) -> GenericVaspOutput:
    executor = ctx.resource_manager.default_executor

    # For the first round
    if not input.initiated:
        input.system_files += ctx.resource_manager.get_artifacts(input.config.init_system_files)

    # setup workspace
    work_dir = os.path.join(executor.work_dir, ctx.path_prefix)
    [tasks_dir] = executor.setup_workspace(work_dir, ['tasks'])

    # prepare input template
    if isinstance(input.config.input_template, str):
        input_template = input.config.input_template
        if input_template.startswith('@'):
            input_template = \
                ctx.resource_manager.resolve_artifact(input_template[1:])[0].url
        input_template = Incar.from_file(input_template).as_dict()
    else:
        input_template = copy.deepcopy(input.config.input_template)

    # prepare potcar
    if isinstance(input.config.potcar_source, dict):
        potcar_source = input.config.potcar_source
    else:
        # default: use same sequence as type_map
        if len(input.config.potcar_source) >= len(input.type_map):
            potcar_source = {
                k: v for k, v in zip(input.type_map, input.config.potcar_source)
            }
        else:
            raise ValueError('potcar_source should not be shorter than type_map')
        
    for k, v in potcar_source.items():
        if v.startswith('@'):
            potcar_source[k] = \
                ctx.resource_manager.resolve_artifact(v[1:])[0].url

    # prepare kpoints
    kpoints_template = input.config.kpoints_template
    if kpoints_template:
        if kpoints_template.startswith('@'):
            logger.info(f'resolve artifact {kpoints_template}')
            kpoints_template = \
                ctx.resource_manager.resolve_artifact(kpoints_template[1:])[0].url
        kpoints_template = Kpoints.from_file(kpoints_template).as_dict()
    else:
        kpoints_template = None

    # resolve data files
    lammps_dump_files: List[Artifact] = []
    xyz_files: List[Artifact] = []

    # TODO: support POSCAR in the future
    # TODO: refactor the way of handling different file formats
    system_files = ctx.resource_manager.resolve_artifacts(input.system_files)
    for system_file in system_files:
        if LammpsOutputHelper.is_match(system_file):
            lammps_out = LammpsOutputHelper(system_file)
            lammps_dump_files.extend(lammps_out.get_passed_dump_files())
        elif XyzHelper.is_match(system_file):
            xyz_files.append(system_file)
        else:
            raise ValueError(f'unsupported format {system_file.url}: {system_file.format}')

    # create task dirs and prepare input files
    vasp_task_dirs = []
    if lammps_dump_files or xyz_files:
        vasp_task_dirs = executor.run_python_fn(make_vasp_task_dirs)(
            lammps_dump_files=[a.to_dict() for a in lammps_dump_files],
            xyz_files=[a.to_dict() for a in xyz_files],
            type_map=input.type_map,
            base_dir=tasks_dir,
            input_template=input_template,
            potcar_source=potcar_source,
            kpoints_template=kpoints_template,
            limit=input.config.limit,
        )
    else:
        logger.warn('no available candidates, skip')
        return GenericVaspOutput(vasp_outputs=[])

    # build commands
    steps = []
    for vasp_task_dir in vasp_task_dirs:
        steps.append(BashStep(
            cwd=vasp_task_dir['url'],
            cmd=[ctx.config.vasp_cmd, ' 1>> output 2>> output'],
            checkpoint='vasp',
        ))

    # run tasks
    jobs = []
    for i, steps_group in enumerate(split_list(steps, ctx.config.concurrency)):
        if not steps_group:
            continue
        script = BashScript(
            template=ctx.config.script_template,
            steps=steps,
        )
        job = executor.submit(script.render(), cwd=tasks_dir,
                              checkpoint_key=f'submit-job/vasp/{i}:{tasks_dir}')
        jobs.append(job)
    jobs = await gather_jobs(jobs, max_tries=2)

    vasp_outputs = [Artifact.of(
        url=a['url'],
        format=VaspOutputHelper.format,
        executor=executor.name,
        attrs=a['attrs'],
    ) for a in vasp_task_dirs]

    return GenericVaspOutput(vasp_outputs=vasp_outputs)


def __make_vasp_task_dirs():
    def make_vasp_task_dirs(lammps_dump_files: List[ArtifactDict],
                            xyz_files: List[ArtifactDict],
                            type_map: List[str],
                            input_template: dict,
                            potcar_source: dict,
                            base_dir: str,
                            kpoints_template: Optional[dict] = None,
                            limit: int = 0
                            ) -> List[ArtifactDict]:
        """Generate VASP input files from LAMMPS dump files or XYZ files."""

        import ase.io
        from ase import Atoms
        from ase.io.vasp import _symbol_count_from_symbols

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

            # create input file
            input_data = copy.deepcopy(input_template)
            input_data = dict_nested_get(
                file, ['attrs', 'vasp', 'input_data'], input_data # type: ignore
            )
            incar = Incar.from_dict(input_data)
            incar.write_file(os.path.join(task_dir, 'INCAR'))

            # create POSCAR
            elements_all = atoms.get_chemical_symbols()
            elements = [
                item[0] for item in _symbol_count_from_symbols(elements_all)
            ]
            ase.io.write(
                os.path.join(task_dir, 'POSCAR'), atoms, format='vasp5'
            )

            # create POTCAR
            with open(os.path.join(task_dir, 'POTCAR'), 'w') as out_file:
                for element in elements:
                    with open(potcar_source[element], 'r') as in_file:
                        out_file.write(in_file.read())

            # create KPOINTS
            kpoints_template = dict_nested_get(
                file, ['attrs', 'vasp', 'kpoints_template'] # type: ignore
            )
            if kpoints_template:
                kpoints = Kpoints.from_dict(kpoints_template)
                kpoints.write_file(os.path.join(task_dir, 'KPOINTS'))

            # inherit attrs from input file
            # TODO: inherit only ancestor key should be enough
            task_dirs.append({
                'url': task_dir,
                'attrs': file['attrs'],
            })

        return task_dirs

    return make_vasp_task_dirs
make_vasp_task_dirs = __make_vasp_task_dirs()
