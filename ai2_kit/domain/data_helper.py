from ai2_kit.core.artifact import Artifact, ArtifactDict

from typing import List, Tuple, Optional
from ase import Atoms
import os
import re

from .constant import MODEL_DEVI_OUT, LAMMPS_TRAJ_DIR, LAMMPS_TRAJ_SUFFIX


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

    def get_passed_dump_files(self) -> List[Artifact]:
        return [self.artifact.join(LAMMPS_TRAJ_DIR, f'{i}{LAMMPS_TRAJ_SUFFIX}') for i in self.artifact.attrs['passed']]

class PoscarHelper(DataHelper):
    format = 'vasp/poscar'
    pattern = re.compile(r'POSCAR')


class XyzHelper(DataHelper):
    format = 'extxyz'
    suffix = '.xyz'

class DeepmdNpyHelper(DataHelper):
    format = 'deepmd/npy'

class DeepmdModelHelper(DataHelper):
    format = 'deepmd/model'

class Cp2kOutputHelper(DataHelper):
    format = 'cp2k-output-dir'

class VaspOutputHelper(DataHelper):
    format = 'vasp-output-dir'



def __ase_atoms_to_cp2k_input_data():
    """workaround for cloudpickle issue"""
    def ase_atoms_to_cp2k_input_data(atoms: Atoms) -> Tuple[List[str], List[List[float]]]:
        """
        Convert ASE Atoms to CP2K input format
        """
        coords = [atom.symbol + ' ' + ' '.join(str(x) for x in atom.position) for atom in atoms] # type: ignore
        cell = [list(row) for row in atoms.cell]  # type: ignore
        return (coords, cell)
    return ase_atoms_to_cp2k_input_data
ase_atoms_to_cp2k_input_data = __ase_atoms_to_cp2k_input_data()


def __convert_to_deepmd_npy():
    """workaround for cloudpickle issue"""
    def covert_to_deepmd_npy(cp2k_outputs: List[ArtifactDict], base_dir: str, type_map: List[str]):
        import dpdata
        from itertools import groupby

        atoms_list: List[Tuple[ArtifactDict, Atoms]] = []
        for cp2k_output in cp2k_outputs:
            dp_system = dpdata.LabeledSystem(os.path.join(cp2k_output['url'], 'output'), fmt='cp2k/output', type_map=type_map)
            atoms_list += [
                (cp2k_output, atoms)
                for atoms in dp_system.to_ase_structure()  # type: ignore
            ]

        output_dirs = []
        # group dataset by ancestor key
        for i, (key, atoms_group) in enumerate(groupby(atoms_list, key=lambda x: x[0]['attrs']['ancestor'])):
            output_dir = os.path.join(base_dir, key.replace('/', '_'))
            atoms_group = list(item[1] for item in atoms_group)
            if not atoms_group:
                continue
            dp_system = dpdata.LabeledSystem(atoms_group[0], fmt='ase/structure')
            for atoms in atoms_group[1:]:
                dp_system += dpdata.LabeledSystem(atoms, fmt='ase/structure')
            dp_system.to_deepmd_npy(output_dir, set_size=len(dp_system))  # type: ignore
            # TODO: return ArtifactDict
            output_dirs.append(output_dir)
        return output_dirs

    return covert_to_deepmd_npy
convert_to_deepmd_npy = __convert_to_deepmd_npy()


def __convert_to_lammps_input_data():
    """workaround for cloudpickle issue"""
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

    return convert_to_lammps_input_data
convert_to_lammps_input_data = __convert_to_lammps_input_data()
