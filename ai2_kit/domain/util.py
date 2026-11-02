import numpy as np
from ase.calculators.lammps import Prism, convert


def cp2k_dumps_input(input_dict):
    # Initialize an empty list for output lines
    output_lines = []
    # Define a helper function to recursively dump the sections and values
    def dump_section(section_dict, indent=0):
        # Loop through the keys and values in the section dictionary
        for key, value in section_dict.items():
            # Check if the value is a sub-dictionary
            if isinstance(value, dict):
                # Add a line with the section name and indentation
                output_lines.append(" " * indent + f"&{key}")
                # Recursively dump the sub-section with increased indentation
                dump_section(value, indent + 3)
                # Add a line with the end of section and indentation
                output_lines.append(" " * indent + "&END")
            else:
                # Add a line with the value name and value and indentation
                output_lines.append(" " * indent + f"{key}  {value}")
    # Dump the input dictionary using the helper function
    dump_section(input_dict)
    # Return the output list as a single string with newline characters
    return "\n".join(output_lines)


def cp2k_dump_input(input_dict, fp):
    fp.write(cp2k_dumps_input(input_dict))


class LammpsData:
    """
    LammpsData is a class to read and write lammps data file.
    from: https://gitee.com/chiahsinchu/toolbox/blob/dev/toolbox/io/lammps.py
    """

    def __init__(self, atoms) -> None:
        self.atoms = atoms
        self._setup()

        self.angles = None
        self.bonds = None
        self.dihedrals = None
        self.velocities = None

    def write(self, fp, **kwargs):
        specorder = kwargs.get("specorder", None)
        if specorder is not None:
            self.set_atype_from_specorder(specorder)
            n_atype = len(specorder)
        else:
            n_atype = len(np.unique(self.atoms.numbers))
        atom_style = kwargs.get("atom_style", "full")

        header = self._make_header(fp.name, n_atype)
        fp.write(header)
        body = self._make_atoms(atom_style)
        fp.write(body)
        if self.bonds is not None:
            fp.write("\nBonds\n\n")
            np.savetxt(fp, self.bonds, fmt="%d")
        if self.angles is not None:
            fp.write("\nAngles\n\n")
            np.savetxt(fp, self.angles, fmt="%d")
        if self.dihedrals is not None:
            fp.write("\nDihedrals\n\n")
            np.savetxt(fp, self.dihedrals, fmt="%d")
        if self.velocities is not None:
            fp.write("\nVelocities\n\n")
            np.savetxt(fp, self.velocities, fmt=["%d", "%.16f", "%.16f", "%.16f"])

    def _make_header(self, out_file, n_atype):
        nat = len(self.atoms)
        s = "%s (written by toolbox by Jia-Xin Zhu)\n\n" % out_file
        s += "%d atoms\n" % nat
        s += "%d atom types\n" % n_atype
        if self.bonds is not None:
            s += "%d bonds\n" % len(self.bonds)
            s += "%d bond types\n" % len(np.unique(self.bonds[:, 1]))
        if self.angles is not None:
            s += "%d angles\n" % len(self.angles)
            s += "%d angle types\n" % len(np.unique(self.angles[:, 1]))
        if self.dihedrals is not None:
            s += "%d dihedrals\n" % len(self.dihedrals)
            s += "%d dihedral types\n" % len(np.unique(self.dihedrals[:, 1]))
        prismobj = Prism(self.atoms.get_cell())
        xhi, yhi, zhi, xy, xz, yz = convert(
            prismobj.get_lammps_prism(), "distance", "ASE", "metal")
        s += "0.0 %.6f xlo xhi\n" % xhi
        s += "0.0 %.6f ylo yhi\n" % yhi
        s += "0.0 %.6f zlo zhi\n" % zhi
        if prismobj.is_skewed():
            s += "%.6f %.6f %.6f xy xz yz\n" % (xy, xz, yz)
        s += "\n"
        return s

    def _make_atoms(self, atom_style):
        """
        full atom_id res_id type q x y z
        atomic ...
        """
        return getattr(self, "_make_atoms_%s" % atom_style)()

    def _make_atoms_full(self):
        """
        full atom_id res_id type q x y z
        """
        s = "Atoms\n\n"
        for atom in self.atoms:
            ii = atom.index
            s += "%d %d %d %.6f %.6f %.6f %.6f\n" % (
                ii + 1, self.res_id[ii], self.atype[ii], self.charges[ii],
                self.positions[ii][0], self.positions[ii][1],
                self.positions[ii][2])
        return s

    def _make_atoms_atomic(self):
        pass

    def set_res_id(self, res_id):
        self.res_id = np.reshape(res_id, (-1))

    def set_atype(self, atype):
        self.atype = np.reshape(atype, (-1))

    def set_atype_from_specorder(self, specorder):
        atype = []
        for ii in self.atoms.get_chemical_symbols():
            atype.append(specorder.index(ii))
        self.atype = np.array(atype, dtype=np.int32) + 1

    def set_bonds(self, bonds):
        self.bonds = np.reshape(bonds, (-1, 4))

    def set_angles(self, angles):
        self.angles = np.reshape(angles, (-1, 5))

    def set_dihedral(self, dihedrals):
        self.dihedrals = np.reshape(dihedrals, (-1, 6))

    def set_velocities(self, velocities):
        self.velocities = np.reshape(velocities, (-1, 4))

    def _setup(self):
        self.positions = self.atoms.get_positions()
        if len(self.atoms.get_initial_charges()) > 0:
            self.charges = self.atoms.get_initial_charges().reshape(-1)
        else:
            self.charges = np.zeros(len(self.atoms))

        if hasattr(self, "res_id"):
            assert len(self.res_id) == len(self.atoms)
        else:
            self.res_id = np.zeros(len(self.atoms), dtype=np.int32)
