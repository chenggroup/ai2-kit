from ase.data import atomic_masses, atomic_numbers

class LammpsTool:

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