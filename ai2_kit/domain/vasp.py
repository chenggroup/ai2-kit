from ai2_kit.core.artifact import Artifact, ArtifactDict
from ai2_kit.core.script import BashScript, BashStep, BashTemplate
from ai2_kit.core.job import gather_jobs
from ai2_kit.core.util import dict_nested_get, split_list, list_even_sample, shuffle_sample
from ai2_kit.core.log import get_logger

from typing import List, Union, Literal
from pydantic import BaseModel
from dataclasses import dataclass

from typing import List, Union, Optional, Tuple
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pydantic import BaseModel
from dataclasses import dataclass

import copy
import os
import random

from .data import LammpsOutputHelper, XyzHelper, DataFormat
from .iface import ICllLabelOutput, BaseCllContext

logger = get_logger(__name__)

class GenericVaspInputConfig(BaseModel):
    init_system_files: List[str] = []
    input_template: Union[dict, str]
    potcar_source: Union[dict, list]
    kpoints_template: Optional[Union[dict, str]] = None
    """
    Input template for VASP. Could be a dict or content of a VASP input file.
    """
    limit: int = 50
    sample_method: Literal["even", "random"] = "even"

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
        input_template = Incar.from_file(input_template).as_dict()
    else:
        input_template = copy.deepcopy(input.config.input_template)

    # prepare potcar
    if isinstance(input.config.potcar_source, dict):
        potcar_source = input.config.potcar_source
    elif isinstance(input.config.potcar_source, list):
        # default: use same sequence as type_map
        if len(input.config.potcar_source) >= len(input.type_map):
            potcar_source = {
                k: v for k, v in zip(input.type_map, input.config.potcar_source)
            }
        else:
            raise ValueError('potcar_source should not be shorter than type_map')
    else:
        # TODO: support generate POTCAR from given path of potential.
        raise ValueError('potcar_source should be either dict or list')

    # prepare kpoints
    kpoints_template = input.config.kpoints_template
    if isinstance(kpoints_template, str):
        kpoints_template = Kpoints.from_file(kpoints_template).as_dict()
    elif isinstance(kpoints_template, dict):
        kpoints_template = copy.deepcopy(kpoints_template)
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
            lammps_selected_dumps = lammps_out.get_selected_dumps()
            lammps_dump_files.extend(lammps_selected_dumps)
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

    # submit tasks and wait for completion
    jobs = []
    for i, steps_group in enumerate(split_list(steps, ctx.config.concurrency)):
        if not steps_group:
            continue
        script = BashScript(
            template=ctx.config.script_template,
            steps=steps_group,
        )
        job = executor.submit(script.render(), cwd=tasks_dir,
                              checkpoint_key=f'queue-job:vasp:{tasks_dir}:{i}')
        jobs.append(job)
    jobs = await gather_jobs(jobs, max_tries=2)

    vasp_outputs = [Artifact.of(
        url=a['url'],
        format=DataFormat.VASP_OUTPUT_DIR,
        executor=executor.name,
        attrs=a['attrs'],
    ) for a in vasp_task_dirs]

    return GenericVaspOutput(vasp_outputs=vasp_outputs)


def __export_remote_functions():
    def make_vasp_task_dirs(lammps_dump_files: List[ArtifactDict],
                            xyz_files: List[ArtifactDict],
                            type_map: List[str],
                            input_template: dict,
                            potcar_source: dict,
                            base_dir: str,
                            kpoints_template: Optional[dict] = None,
                            limit: int = 0,
                            sample_method: Literal["even", "random"] = "even"
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
            if sample_method == "even":
                atoms_list = list_even_sample(atoms_list, limit)
            elif sample_method == "random":
                atoms_list = shuffle_sample(atoms_list, limit)
            else:
                raise ValueError(f"Unknown sample method {sample_method}")

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
                os.path.join(task_dir, 'POSCAR'), atoms, format='vasp'
            )

            # create POTCAR
            with open(os.path.join(task_dir, 'POTCAR'), 'w') as out_file:
                for element in elements:
                    with open(potcar_source[element], 'r') as in_file:
                        out_file.write(in_file.read())

            # create KPOINTS
            kpoints_template = dict_nested_get(
                file, ['attrs', 'vasp', 'kpoints_template'], None # type: ignore
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
make_vasp_task_dirs = __export_remote_functions()
