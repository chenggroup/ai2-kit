from typing import Optional
from ase.data import atomic_masses, atomic_numbers
import pandas as pd
import json

class LammpsTool:
    """
    A tool for LAMMPS related operations.
    """

    def analysis_density_convergence(self, in_file: str, out_file: Optional[str] = None,
                                     debug=False, err=0.0005, col_name: str = 'density'):
        """
        Select a frame by density from a property file.

        :param in_file: input properties file, e.g. properties.out
        :param out_file: output file to save the selected frame, if None, return the result
        :param col_name: column name to select by, default is 'density'
        :param debug: if True, return the average density for each frame
        :param err: error threshold for convergence, default is 0.0005

        :return: a dictionary with converged density and frames, or save to out_file
        """
        with open(in_file, 'r') as f:
            f.seek(1)  # skip the leading '#'
            df = pd.read_csv(f, delim_whitespace=True, header=0)
        if col_name not in df.columns:
            raise ValueError(f'Column {col_name} not found in {in_file}')
        density = df[col_name]

        avg_density = density.cumsum() / (df.index + 1)
        converged_density = avg_density.iloc[-1]
        converged_df = df[(density - converged_density).abs() < err]

        result = {
            'converged_density': converged_density,
            'converged_frames': converged_df.index.tolist(),
        }
        if debug:
            result['time'] = df['time'].tolist()
            result['avg_density'] = avg_density.tolist()

        if out_file:
            with open(out_file, 'w') as f:
                json.dump(result, f)
        else:
            return result

    def gen_mass_map(self, atom_types: tuple, out_file = None):
        """
        Generate a mass map for LAMMPS.
        :param atom_types: tuple of atom types, e.g. ('C', 'H', 'O')
        :return: LAMMPS mass map string
        :param out_file: output file to save the mass map, if None, return the string

        :return: mass map string, e.g.:
        ```
        mass 1 12.011 # C
        mass 2 1.008 # H
        mass 3 15.999 # O
        ```
        """
        mass_list = []
        # lookup mass from ase
        for i, atom_type in enumerate(atom_types, start=1):
            atom_no = atomic_numbers[atom_type]
            if atom_no is None:
                raise ValueError(f'Unknown atom type: {atom_type}')
            mass = atomic_masses[atom_no]
            mass_list.append(f'mass {i:>4} {mass:>8}  # {atom_type}')
        mass_map = '\n'.join(mass_list)
        if out_file:
            with open(out_file, 'w') as f:
                f.write(mass_map)
        return mass_map
