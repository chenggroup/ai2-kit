from typing import List, Tuple
from ase import Atoms


def ase_atoms_to_cp2k_input_data(atoms: Atoms) -> Tuple[List[str], List[List[float]]]:
    coords = [atom.symbol + ' ' + ' '.join(str(x) for x in atom.position) for atom in atoms] # type: ignore
    cell = [list(row) for row in atoms.cell]  # type: ignore
    return (coords, cell)
