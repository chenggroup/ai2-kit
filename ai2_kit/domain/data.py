from ai2_kit.core.artifact import Artifact, ArtifactDict

from enum import Enum
from typing import List, Tuple, Optional
from ase import Atoms
import os
import re

from .constant import LAMMPS_TRAJ_DIR, LAMMPS_TRAJ_SUFFIX, LAMMPS_DUMPS_CLASSIFIED


class DataHelper:

    format: Optional[str] = None
    suffix: Optional[str] = None
    pattern: Optional[re.Pattern] = None

    @classmethod
    def is_match(cls, artifact: Artifact) -> bool:
        if cls.format and artifact.format == cls.format:
            return True
        if cls.suffix and artifact.url.endswith(cls.suffix):
            return True
        file_name = os.path.basename(artifact.url)
        if cls.pattern and cls.pattern.match(file_name):
            return True
        return False

    def __init__(self, artifact: Artifact) -> None:
        self.artifact = artifact


class LammpsOutputHelper(DataHelper):
    format = 'lammps/output-dir'

    def get_model_devi_file(self, filename: str) -> Artifact:
        return self.artifact.join(filename)

    def get_selected_dumps(self) -> List[Artifact]:
        dumps = []
        for selected_dump_id in self.artifact.attrs[LAMMPS_DUMPS_CLASSIFIED]['selected']:
            dump = self.artifact.join(LAMMPS_TRAJ_DIR, f'{selected_dump_id}{LAMMPS_TRAJ_SUFFIX}')
            dump.attrs = { **self.artifact.attrs, LAMMPS_DUMPS_CLASSIFIED: None}
            dumps.append(dump)
        return dumps


class PoscarHelper(DataHelper):
    format = 'vasp/poscar'
    pattern = re.compile(r'POSCAR')


class XyzHelper(DataHelper):
    format = 'extxyz'
    suffix = '.xyz'


def __export_remote_functions():
    """workaround for cloudpickle issue"""

    class DataFormat:
        # customize data format
        CP2K_OUTPUT_DIR = 'cp2k/output_dir'
        VASP_OUTPUT_DIR = 'vasp/output_dir'
        LAMMPS_OUTPUT_DIR = 'lammps/output_dir'
        DEEPMD_OUTPUT_DIR = 'deepmd/output_dir'
        DEEPMD_NPY = 'deepmd/npy'

        # data format of dpdata
        CP2K_OUTPUT = 'cp2k/output'
        VASP_XML = 'vasp/xml'

        # data format of ase
        EXTXYZ = 'extxyz'
        VASP_POSCAR = 'vasp/poscar'


    def get_data_format(artifact: dict) -> Optional[str]:
        url = artifact.get('url')
        assert isinstance(url, str), f'url must be str, got {type(url)}'

        file_name = os.path.basename(url)
        format = artifact.get('format')
        if format and isinstance(format, str):
            return format  # TODO: validate format
        if file_name.endswith('.xyz'):
            return DataFormat.EXTXYZ
        if 'POSCAR' in file_name:
            return DataFormat.VASP_POSCAR
        return None


    def ase_atoms_to_cp2k_input_data(atoms: Atoms) -> Tuple[List[str], List[List[float]]]:
        coords = [atom.symbol + ' ' + ' '.join(str(x) for x in atom.position) for atom in atoms] # type: ignore
        cell = [list(row) for row in atoms.cell]  # type: ignore
        return (coords, cell)


    def convert_to_deepmd_npy(
        base_dir: str,
        type_map: List[str],
        dataset: List[ArtifactDict],
    ):
        import dpdata
        from itertools import groupby

        dp_system_list: List[Tuple[ArtifactDict, dpdata.LabeledSystem]]= []
        for data in dataset:
            data_format = get_data_format(data)  # type: ignore
            dp_system = None
            try:
                if data_format == DataFormat.CP2K_OUTPUT_DIR:
                    dp_system = dpdata.LabeledSystem(os.path.join(data['url'], 'output'), fmt='cp2k/output', type_map=type_map)
                elif data_format == DataFormat.VASP_OUTPUT_DIR:
                    dp_system = dpdata.LabeledSystem(os.path.join(data['url'], 'vasprun.xml'), fmt='vasp/xml', type_map=type_map)
            except Exception as e:
                print(f'failed to load {data["url"]}: {e}')

            if dp_system is not None:
                dp_system_list.append((data, dp_system))

        output_dirs = []
        # group dataset by ancestor key
        for i, (key, dp_system_group) in enumerate(groupby(dp_system_list, key=lambda x: x[0]['attrs']['ancestor'])):
            dp_system_group = list(dp_system_group)
            if 0 == len(dp_system_group):
                continue  # skip empty dataset

            # merge dp_systems with the same ancestor into single data set
            output_dir = os.path.join(base_dir, key.replace('/', '_'))
            dp_system = sum([x[1] for x in dp_system_group])
            try:
                dp_system.to_deepmd_npy(output_dir, set_size=len(dp_system), type_map=type_map)  # type: ignore
            except Exception as e:
                print(f'failed to convert {key}: {e}')
                continue
            # inherit attrs key from input artifact
            output_dirs.append({'url': output_dir,
                                'format': DataFormat.DEEPMD_NPY,
                                'attrs': dp_system_group[0][0]['attrs']})  # type: ignore
        return output_dirs


    def convert_to_lammps_input_data(poscar_files: List[ArtifactDict], base_dir: str, type_map: List[str]):
        import dpdata
        import os
        lammps_data_files = []
        for i, poscar_file in enumerate(poscar_files):
            output_file = os.path.join(base_dir, f'{i:06d}.lammps.data')
            dpdata.System(poscar_file['url'], fmt='vasp/poscar', type_map=type_map).to_lammps_lmp(output_file)  # type: ignore
            lammps_data_files.append({
                'url': output_file,
                'attrs': poscar_file['attrs'],
            })
        return lammps_data_files

    return (
        ase_atoms_to_cp2k_input_data,
        convert_to_deepmd_npy,
        convert_to_lammps_input_data,
        DataFormat,
        get_data_format,
    )

(
    ase_atoms_to_cp2k_input_data,
    convert_to_deepmd_npy,
    convert_to_lammps_input_data,
    DataFormat,
    get_data_format,
) = __export_remote_functions()
