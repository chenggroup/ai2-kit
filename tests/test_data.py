from unittest import TestCase, skip
from pathlib import Path

import dpdata
from ase.io import read
from ase import Atom, Atoms
import numpy as np
from typing import List

data_dir = Path(__file__).parent / 'data-sample'


class TestDataTransform(TestCase):
    lammps_dump_file = data_dir / 'h2o.lammps.dump'
    xyz_file = data_dir / 'h2o.xyz'
    cp2k_output_file = data_dir / 'h2o.cp2k.output'

    type_map = ['O', 'H']

    coords_0 = 'H 0.613889 5.74221 0.851211'
    cell = [[12.4200000763, 0.0, 0.0], [0.0, 12.4200000763, 0.0], [0.0, 0.0, 12.4200000763]]

    def test_dpdata_lammps_to_cp2k_input(self):
        dpdata_system = dpdata.System(self.lammps_dump_file, fmt='lammps/dump', type_map=self.type_map)

        atom_names = np.array(dpdata_system['atom_names'])  # type: ignore
        atom_types = dpdata_system['atom_types']  # type: ignore
        coord_arr = dpdata_system['coords'][0]  # type: ignore
        kind_arr = atom_names[atom_types]  # type: ignore
        coords = [ str(k) + ' ' + ' '.join(str(x) for x in c)  for k, c in zip(kind_arr, coord_arr) ] # type: ignore

        # format cell
        cell = dpdata_system['cells'][0]
        cell = np.reshape(cell, [3,3])  # type: ignore

        self.assertEqual(coords[0], self.coords_0)
        cell = [
            list(cell[0, :]),
            list(cell[1, :]),
            list(cell[2, :]),
        ]
        self.assertEqual(str(cell), str(self.cell))

    def test_ase_lammps_to_cp2k_input(self):
        atoms_list: List[Atoms] = read(self.lammps_dump_file, ':', format='lammps-dump-text', order=False, specorder=self.type_map)  # type: ignore
        atoms: Atoms = atoms_list[0]
        coords = [atom.symbol + ' ' + ' '.join(str(x) for x in atom.position) for atom in atoms]  # type: ignore
        self.assertEqual(coords[0], self.coords_0)

        cell = [ list(row) for row in atoms.cell ]  # type: ignore
        self.assertEqual(str(cell), str(self.cell))

    def test_dpdata_to_ase(self):
        dp_system = dpdata.LabeledSystem(self.cp2k_output_file, fmt='cp2k/output')

        atoms_list: List[Atoms] = dp_system.to_ase_structure()  # type: ignore
        dp_system = dpdata.LabeledSystem(atoms_list[0], fmt='ase/structure')
        dp_system += dpdata.LabeledSystem(atoms_list[0], fmt='ase/structure')


class TestDataFile(TestCase):
    plumed_colvar_file = data_dir / 'plumed_colvar.txt'

    def test_colvar(self):
        from ai2_kit.lib import plumed

        df1 = plumed.load_colvar_from_files(self.plumed_colvar_file)
        df2 = plumed.load_colvar_from_files(self.plumed_colvar_file, self.plumed_colvar_file)

        self.assertEqual(len(df1), 20)
        self.assertEqual(len(df2), 40)


        cvs, bias = plumed.get_cvs_bias_from_df(df1, ['d1'], 'opes.bias')
        self.assertEqual(cvs.shape, (20,))
        grid = np.linspace(cvs.min(), cvs.max(),100)
        self.assertEqual(grid.shape, (100,))


        cvs, bias = plumed.get_cvs_bias_from_df(df1, ['d1', 'd2'], 'opes.bias')
        self.assertEqual(cvs.shape, (2, 20))

        grid = np.linspace(cvs.min(axis=1), cvs.max(axis=1),100).T
        self.assertEqual(grid.shape, (2, 100))

