from ase import Atoms
from typing import Iterable

from .data import ase_atoms_to_cp2k_input_data
from .util import cp2k_dump_input


def dump_coord_n_cell(fp, atoms: Atoms):
    coords, cell = ase_atoms_to_cp2k_input_data(atoms)
    cp2k_dump_input({
        'COORD': dict.fromkeys(coords, ''),  # FIXME: this is a dirty hack, should make dump_cp2k_input support COORD
        # use fp32 precision, or 7 decimal places
        'CELL': {
            'A': ' '.join( f'{str(v)[:8]}' for v in cell[0]),
            'B': ' '.join( f'{str(v)[:8]}' for v in cell[1]),
            'C': ' '.join( f'{str(v)[:8]}' for v in cell[2]),
        }
    }, fp)


def lammps_efield_to_cp2k(efield: Iterable[float]):
    """
    Convert a LAMMPS electric field vector to CP2K intensity and polarization.
    """
    import numpy as np
    from scipy import constants

    efield = np.array(efield)
    factor = constants.physical_constants["atomic unit of electric field"][0] * constants.angstrom
    intensity = np.linalg.norm(efield)
    if intensity == 0:
        polarization = np.array([0.0, 0.0, 0.0])
    else:
        polarization = efield / np.linalg.norm(efield)
    return intensity / factor, polarization  # type: ignore
