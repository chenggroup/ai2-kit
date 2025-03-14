# Model Deviation Toolkit

This toolkit is a command line tool to filter structures by model deviation.
This toolkit is part of ase toolkit, you can use it to process trajectory via command line.

## Usage
TODO

## Examples

### Filter LAMMPS trajectory by model deviation

When you run multiple LAMMPS simulations with deepmd-kit, you may want to filter out the structures with model deviation for the next round. Consider the following example:

* There are several LAMMPS jobs under path: `./workdir/lammps/*`
* The trajectory of each job is stored in `./workdir/lammps/*/dump.lammpstrj`
* The model deviation of each job is stored in `./workdir/lammps/*/model_devi.out`
* You want to filter out the structures whose max_devi_f is between 0.1 and 0.2

Then you can use the following command to filter out the structures with model deviation:

```bash
ai2-kit tool model_devi read "./workdir/lammps/*" --traj_file dump.lammpstrj --md_file model_devi.out \
  - grade --lo 0.1 --hi 0.2 --col max_devi_f \
  - dump_stats stats.tsv \
  - write decent.xyz --level decent
```

Explanation of the command:

* `ai2-kit tool model_devi read "./workdir/lammps/*" --traj_file dump.lammpstrj --md_file model_devi.out`: read all LAMMPS trajectory and model deviation files

* `- grade --lo 0.1 --hi 0.2 --col max_devi_f`: grade the structures by `max_devi_f` column, and grade the structures into 3 levels: `good`, `decent`, and `poor`. The structures with `max_devi_f` between 0.1 and 0.2 are graded as `decent`, and the structures with `max_devi_f` less than 0.1 are graded as `good`, and the structures with `max_devi_f` greater than 0.2 are graded as `poor`.

* `- dump_stats stats.tsv`: dump the statistics of the grading process to `stats.tsv`

* `- write decent.xyz --level decent`: write the structures with grade `decent` to `decent.xyz`
