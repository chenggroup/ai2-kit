# Batch Script Helper
```bash
ai2-kit tool batch --help
```

A toolkit to generate batch scripts.

## Usage

```bash

ai2-kit tool batch map_path --help
ai2-kit tool batch run_cmd --help
ai2-kit tool batch gen_batches --help
```

## Examples

### Generate batch scripts to run CP2K tasks

Suppose you want to run new CP2K tasks based on the previous runs, whose work directories can be found with `./old/iters*/label-cp2k/tasks/*`, and the new CP2K config file is in `./new/cp2k.inp`, then you can do the following steps:

#### 1. Create new tasks directories by linking to (or copying) the old ones

```bash
ai2-kit tool batch map_path ./old/iters*/label-cp2k/tasks/* --target "new/task-{i:04d}"
```
The command above will create new directories `new/task-0000`, `new/task-0001`, etc. and link the old tasks directories to them. If you want to copy the old tasks directories instead of linking them, you can use `--copy` option.

#### 2. Generate batch scripts

Suppose you are using SLURM as your job scheduler, then you need to prepare a header file for your batch scripts. For example, you can create a file named `header.sh` with the following content:

```bash
#!/bin/bash
#SBATCH --job-name=cp2k
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32

module load intel/2019.5
module load cp2k/7.1
```

Suppose the command you want to run in each task directory is `mpirun cp2k.popt -i ../cp2k.inp &> output` 
(note that the command is run in the task directory, so the input file is at `../cp2k.inp`),
then you can generate batch scripts with the following command:

```bash
ai2-kit tool batch gen_batches new/task-* --header_file header.sh --cmd "mpirun cp2k.popt -i ../cp2k.inp &> output" \
    --out_script "new/cp2k-{i}.sbatch" --concurrency 5
```

The above command will generate 5 batch scripts named `new/cp2k-0.sbatch`, `new/cp2k-1.sbatch`, etc. and each batch script will run part of the tasks in `new/task-*` directory. 

#### 3. Submit batch scripts to your job scheduler
You can submit the batch scripts to your job scheduler with `sbatch` and `xargs` commands.
```bash
ls -1 new/cp2k-*.sbatch | xargs -I {} sbatch {}
```

## Tips

* You can use the `write_each_frames` in `ase tool` to generate multiple input files, then use the above command to generate batch scripts for each input file.
