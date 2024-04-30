from ai2_kit.core.artifact import ArtifactDict, Artifact

from typing import List, Tuple, Optional
from ase import Atoms

import ase.io
import os


class DataFormat:
    # customize data format
    CP2K_OUTPUT_DIR = 'cp2k/output_dir'
    VASP_OUTPUT_DIR = 'vasp/output_dir'
    LAMMPS_OUTPUT_DIR = 'lammps/output_dir'
    DEEPMD_OUTPUT_DIR = 'deepmd/output_dir'
    ANYWARE_OUTPUT_DIR = 'anyware/output_dir'

    DEEPMD_MODEL = 'deepmd/model'
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
    """
    Read ase atoms list from artifacts
    Deprecated since it is not recommended to use ArtifactDict
    """
    results = []
    for a in artifacts:
        data_format = get_data_format(a)  # type: ignore
        url = a['url']
        if data_format in [DataFormat.VASP_POSCAR, 'vasp']:
            atoms_list = ase.io.read(url, ':', format='vasp')
        elif data_format in [DataFormat.EXTXYZ, 'extxyz']:
            atoms_list = ase.io.read(url, ':', format='extxyz')
        elif data_format is not None:
            atoms_list = ase.io.read(url, ':', format=data_format)
        else:
            raise ValueError(f'unsupported data format: {data_format}')
        results.extend((a, atoms) for atoms in atoms_list)
    return results


def artifacts_to_ase_atoms_v2(artifacts: List[Artifact]) -> List[Tuple[Artifact, Atoms]]:
    results = []
    for a in artifacts:
        data_format = get_data_format(a.to_dict())  # type: ignore
        url = a.url
        if data_format in [DataFormat.VASP_POSCAR, 'vasp']:
            atoms_list = ase.io.read(url, ':', format='vasp')
        elif data_format in [DataFormat.EXTXYZ, 'extxyz']:
            atoms_list = ase.io.read(url, ':', format='extxyz')
        elif data_format is not None:
            atoms_list = ase.io.read(url, ':', format=data_format)
        else:
            raise ValueError(f'unsupported data format: {data_format}')
        results.extend((a, atoms) for atoms in atoms_list)
    return results


def ase_atoms_to_cp2k_input_data(atoms: Atoms) -> Tuple[List[str], List[List[float]]]:
    coords = [atom.symbol + ' ' + ' '.join(str(x) for x in atom.position) for atom in atoms] # type: ignore
    cell = [list(row) for row in atoms.cell]  # type: ignore
    return (coords, cell)


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
