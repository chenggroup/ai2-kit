# ASE Toolkit

```bash
ai2-kit tool ase
```
This toolkit is a command line wrapper of [ASE](https://wiki.fysik.dtu.dk/ase/) to allow user to process trajectory via command line.

## Usage

This toolkit include the following commands:

| Command | Description | Example | Reference |
| --- | --- | --- | --- |
| read | Read a trajectory file into memory. This command by itself is useless, you should chain other command after reading data into memory. | `ai2-kit tool ase read ./path/to/traj.xyz` | [ase.io.read](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read) |
| write | Write all frame of a trajectory into a single file. | `ai2-kit tool ase read ./path/to/traj.xyz - write ./path/to/output.xyz` | [ase.io.write](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.write) |
| write_each_frame | Write each frame of a trajectory into a separated file. The file name should include `{i}` or other valid Python `str.format` to indicate the frame number | `ai2-kit tool ase read ./path/to/traj.xyz - write_each_frame ./path/to/POSCAR-{i:04d} --format vasp` | [ase.io.write](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.write) |
| set_cell | Set the cell of all frames in the trajectory. | `ai2-kit tool ase read ./path/to/traj.xyz - set_cell "[10,10,10,90,90,90]"` | [ase.Atoms.set_cell](https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.set_cell) |
| set_pbc | Set the periodic boundary condition of all frames in the trajectory. | `ai2-kit tool ase read ./path/to/traj.xyz - set_pbc "[True,True,True]"` | [ase.Atoms.set_pbc](https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.set_pbc) |  

Those commands are chainable and can be used to process trajectory in a pipeline fashion (separated by `-`). For more information, please refer to the following examples.

## Example

```bash
# Convert every frame in xyz trajectory to a POSCAR file
ai2-kit tool ase read ./path/to/traj.xyz - write_each_frame "POSCAR-{i:04d}" --format vasp

# Convert every 20th frame in xyz trajectory to a POSCAR file
# For more information about the index syntax, please refer to https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read
ai2-kit tool ase read ./path/to/traj.xyz --index '::20' - write_each_frame "POSCAR-{i:04d}" --format vasp 

# Convert lammps dump data to a POSCAR file
ai2-kit tool ase read ./path/to/lammp-dump.data --format lammps-dump-text --noorder --specorder "['O','H']" - write "POSCAR" --format vasp 
```
