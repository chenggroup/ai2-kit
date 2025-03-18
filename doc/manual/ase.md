# ASE Toolkit

```bash
ai2-kit tool ase
```
This toolkit is a command line wrapper of [ASE](https://wiki.fysik.dtu.dk/ase/) to allow user to process trajectory via command line.

## Usage

```bash
ai2-kit tool ase  # show all commands
ai2-kit tool ase to_dpdata -h  # show doc of specific command
```

This toolkit include the following commands:

| Command | Description | Example | Reference |
| --- | --- | --- | --- |
| read | Read trajectory files into memory. This command by itself is useless, you should chain other command after reading data into memory. | `ai2-kit tool ase read ./path/to/traj.xyz` | [ase.io.read](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read), support wildcard, can be call multiple times |
| write | Write all frame of a trajectory into a single file. | `ai2-kit tool ase read ./path/to/traj.xyz - write ./path/to/output.xyz` | [ase.io.write](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.write) |
| write_frames | Write each frame of a trajectory into a separated file. The file name should include `{i}` or other valid Python `str.format` to indicate the frame number | `ai2-kit tool ase read ./path/to/traj.xyz - write_frames ./path/to/POSCAR-{i:04d} --format vasp` | [ase.io.write](https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.write) |
| set_cell | Set the cell of all frames in the trajectory. | see in `Example` | [ase.Atoms.set_cell](https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.set_cell) |
| set_pbc | Set the periodic boundary condition of all frames in the trajectory. | see in `Example` | [ase.Atoms.set_pbc](https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.set_pbc) |  
| delete_atoms | Delete atoms from all frames in the trajectory. | see in `Example` | |
| write_dplr_lammps_data | Write data in the format LAMMPS data for DPLR | see in `Example` | |
| slice | use slice expression to process systems | see in `Example` | |
| sample | sample data by different methods, current supported method are `even` and `random` | see in `Example` | |
| to_dpdata | convert ase.Atoms to dpdata and use [dpdata tool](./dpdata.md) to process | see in `Example` |  |

Those commands are chainable and can be used to process trajectory in a pipeline fashion (separated by `-`). For more information, please refer to the following examples.


## Example

```bash
# Convert every frame in xyz trajectory to separated POSCAR files
ai2-kit tool ase read ./path/to/traj.xyz - write_frames "POSCAR-{i:04d}" --format vasp

# Convert every 20th frame in xyz trajectory to separated POSCAR files
# For more information about the index syntax, please refer to https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read
ai2-kit tool ase read ./path/to/traj.xyz --index '::20' - write_frames "POSCAR-{i:04d}" --format vasp 

# Convert a single lammps dump data to a POSCAR file
ai2-kit tool ase read ./path/to/lammp-dump.data --format lammps-dump-text --specorder [O,H] - write POSCAR --format vasp

# Delete atoms from a trajectory
ai2-kit tool ase read lammps.data --format lammps-data --style atomic - delete_atoms [10,12] - write lammps-fin.data --format lammps-data

# Read multiple files and write them into a single file
ai2-kit tool ase read ./path/to/data1/*.xyz - read ./path/to/data2/*.xyz - write all.xyz

# Read all `good` structures generate by CLL training workflow use glob
ai2-kit tool ase read ./workdir/iters-*/selector*/model-devi/*/good.xyz  - write all-good.syz

# Convert xyz file to cp2k-inc file
ai2-kit tool ase read coord.xyz - set_cell "[10,10,10,90,90,90]" - write coord_n_cell.inc --format cp2k-inc

# Convert xyz file to DPLR LAMMPS data
# Note: don't have space in the list or else you have to quote it with ""
ai2-kit tool ase read h2o.xyz - write_dplr_lammps_data tmp/dplr/{i}.lammps.data --type_map [O,H] --sel_type [0] --sys_charge_map [6,1] --model_charge_map [-8]

# Drop the first 10 frames and then sample 10 frames use random method, and save it as dpdata.System format
ai2-kit tool ase read h2o.lammpstrj --specorder [H,O] - slice 10: - sample 10 --method random - to_dpdata - write dp-h2o --nomerge

# Split the trajectory into multiple parts, for example, pick 10 frames randomly and save it to 10.xyz, and the rest to rest.xyz
ai2-kit tool ase read all.xyz - shuffle - write 10.xyz --slice :10 --chain - write rest.xyz --slice 10: 
```
