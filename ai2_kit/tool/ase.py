import ase.io
from ase import Atoms
from typing import List

class AseHelper:
    def __init__(self):
        self._atoms_list: List[Atoms] = []

    def read(self, filename: str, **kwargs):
        kwargs.setdefault('index', ':')
        data = ase.io.read(filename, **kwargs)
        if not isinstance(data, list):
            data = [data]
        self._atoms_list = data
        return self

    def write(self, filename: str, **kwargs):
        ase.io.write(filename, self._atoms_list, **kwargs)

    def set_cell(self, cell, scale_atoms=False, apply_constraint=True):
        for atoms in self._atoms_list:
            atoms.set_cell(cell, scale_atoms=scale_atoms, apply_constraint=apply_constraint)
        return self

    def set_pbc(self, pbc):
        print(pbc)
        for atoms in self._atoms_list:
            atoms.set_pbc(pbc)
        return self

    def write_each_frame(self, filename: str, **kwargs):
        for i, atoms in enumerate(self._atoms_list):
            ase.io.write(filename.format(i=i), atoms, **kwargs)
