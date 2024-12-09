# Model Deviation Toolkit

This toolkit is a command line tool to filter structures by model deviation.
This toolkit is part of ase toolkit, you can use it to process trajectory via command line.

## Usage
TODO

## Examples

### Filter LAMMPS trajectory by model deviation

When you run multiple LAMMPS simulations with deepmd-kit, you may want to filter out the structures with model deviation for the next round. Consider the following example:

* There are several LAMMPS jobs under path: `./workdir/lammps/*`
* The trajectory of each job is stored in `./workdir/lammps/*/traj/*.lammpstrj`
* The model deviation of each job is stored in `./workdir/lammps/*/model_devi.out`
* You want to filter out the structures whose max_devi_f is between 0.1 and 0.2

Then you can use the following command to filter out the structures with model deviation:

```bash
ai2-kit tool ase read "./workdir/lammps/*/traj/*.lammpstrj" --nat_sort \
  - to_model_devi "./workdir/lammps/*/model_devi.out" \
  - grade --lo 0.1 --hi 0.2 --col max_devi_f \
  - dump_stats stats.tsv \
  - to_ase --level bad \
  - write bad.xyz
```

Explanation of the command:
* The double quotes are required to prevent shell expansion of `*`. You must use double quotes with `ase read` and `to_model_devi` to avoid undetermined behavior.
* The `--nat_sort` is used to sort the trajectory by the number of atoms in each frame. It's used to ensure the order of the trajectory is consistent with the model deviation file. 
* The `to_model_devi` command is used to read the model deviation file. Note that the path of the model deviation file must be consistent with the trajectory file.
* The `grade` command is used to grade the structures by the model deviation. The `--lo` and `--hi` are used to set the lower and upper bound of the model deviation. The structures whose  `max_devi_f` is below lo, between lo and hi, and above hi are graded as `good`, `bad`, and `ugly`, respectively.
* The `dump_stats` command is used to dump the statistics of the grading to a file. The statistics include the number of structures in each grade.
 * The `to_ase` command is used to hand over selected structures back to `ase toolkit`. The `--level` is used to select the grade of the structures. The `bad` level is used to select the structures graded as `bad`.
* The `write` command is used to write the selected structures to a file. The `bad.xyz` is the output file name.