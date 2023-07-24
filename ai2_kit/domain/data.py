from ai2_kit.core.artifact import ArtifactDict

from typing import List, Tuple, Optional
from ase import Atoms

import ase.io
import os


def __export_remote_functions():
    """workaround for cloudpickle issue"""

    class DataFormat:
        # customize data format
        CP2K_OUTPUT_DIR = 'cp2k/output_dir'
        VASP_OUTPUT_DIR = 'vasp/output_dir'
        LAMMPS_OUTPUT_DIR = 'lammps/output_dir'
        DEEPMD_OUTPUT_DIR = 'deepmd/output_dir'
        DEEPMD_NPY = 'deepmd/npy'
        LASP_LAMMPS_OUT_DIR ='lasp+lammps/output_dir'

        # data format of dpdata
        CP2K_OUTPUT = 'cp2k/output'
        VASP_XML = 'vasp/xml'

        # data format of ase
        EXTXYZ = 'extxyz'
        VASP_POSCAR = 'vasp/poscar'


    def get_data_format(artifact: dict) -> Optional[str]:
        """
        Get (or guess) data type from artifact dict
        Note: The reason of using dict instead of Artifact is Artifact is not pickleable
        """
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


    def artifacts_to_ase_atoms(artifacts: List[ArtifactDict], type_map: List[str]) -> List[Tuple[ArtifactDict, Atoms]]:
        results = []
        for a in artifacts:
            data_format = get_data_format(a)  # type: ignore
            url = a['url']
            if data_format == DataFormat.VASP_POSCAR:
                atoms_list = ase.io.read(url, ':', format='vasp')
            elif data_format == DataFormat.EXTXYZ:
                atoms_list = ase.io.read(url, ':', format='extxyz')
            else:
                raise ValueError(f'unsupported data format: {data_format}')
            results.extend((a, atoms) for atoms in atoms_list)
        return results


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
                    dp_system = dpdata.LabeledSystem(os.path.join(data['url'], 'OUTCAR'), fmt='vasp/outcar', type_map=type_map)
            except Exception as e:
                print(f'failed to load {data["url"]}: {e}')

            # one case of len(dp_system) == 0 is when the system is not converged
            if dp_system is not None and len(dp_system) > 0:
                dp_system_list.append((data, dp_system))

        output_dirs = []
        # group dataset by ancestor key
        for i, (key, dp_system_group) in enumerate(groupby(dp_system_list, key=lambda x: x[0]['attrs']['ancestor'])):
            dp_system_group = list(dp_system_group)
            if 0 == len(dp_system_group):
                continue  # skip empty dataset

            output_dir = os.path.join(base_dir, key.replace('/', '_'))
            # merge dp_systems with the same ancestor into single data set
            dp_system = dp_system_group[0][1]
            for item in dp_system_group[1:]:
                dp_system += item[1]

            dp_system.to_deepmd_npy(output_dir, set_size=len(dp_system), type_map=type_map)  # type: ignore
            # inherit attrs key from input artifact
            output_dirs.append({'url': output_dir,
                                'format': DataFormat.DEEPMD_NPY,
                                'attrs': dp_system_group[0][0]['attrs']})  # type: ignore
        return output_dirs


    def convert_to_lammps_input_data(systems: List[ArtifactDict], base_dir: str, type_map: List[str]):
        data_files = []
        atoms_list = artifacts_to_ase_atoms(systems, type_map=type_map)
        for i, (artifact, atoms) in enumerate(atoms_list):
            data_file = os.path.join(base_dir, f'{i:06d}.lammps.data')
            ase.io.write(data_file, atoms, format='lammps-data', specorder=type_map)  # type: ignore
            data_files.append({
                'url': data_file,
                'attrs': artifact['attrs'],
            })
        return data_files

    return (
        artifacts_to_ase_atoms,
        ase_atoms_to_cp2k_input_data,
        convert_to_deepmd_npy,
        convert_to_lammps_input_data,
        DataFormat,
        get_data_format,
    )

(
    artifacts_to_ase_atoms,
    ase_atoms_to_cp2k_input_data,
    convert_to_deepmd_npy,
    convert_to_lammps_input_data,
    DataFormat,
    get_data_format,
) = __export_remote_functions()
