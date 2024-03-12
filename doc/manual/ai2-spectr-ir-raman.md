# Use `ai2-spectr` to Predict IR and Raman Spectrum on HPC
```bash
ai2-kit feat spectr viber
```

## Introduction
The `ai2-spectr` toolkit is a comprehensive suite tailored for the prediction and analysis of spectra. The `viber` toolkit, embedded within, specializes in the prediction of vibrational spectra, encompassing both infrared and Raman spectra. Unlike furnishing a fully integrated computational workflow, the toolkit presents an array of tools for job generation, dataset creation, data visualization, and similar functions. Users can effortlessly leverage these tools to construct workflows operational on High-Performance Computing (HPC) systems. This documentation is intended to elucidate the usage of the toolkit, employing concrete examples for a thorough understanding.

## Case Study

### Build a shell workflow to generate dipole and polarizability dataset

This case study illustrates the construction of a workflow using a straightforward shell script. The workflow, driven by input system and CP2K configuration files, autonomously generates jobs. These jobs are then submitted for execution on an HPC cluster, facilitating the production of the dataset essential for training in Deepmd-kit.

To get started, you need to prepare the following files:

* System files for CP2K: They can be any format that supported by `ase.io`, for example `Al3O2.xyz`.
* CP2K input files: The CP2K input files for the system. You need to prepare a configuration for dipole calculation and 3 configurations for polarizability calculation. For example, `cp2k-dipole.inp` and `cp2k-polar-x.inp`, `cp2k-polar-y.inp`, `cp2k-polar-z.inp`.
  * Note that you need to ensure the file names of the wannier output files are different in different CP2K configurations. A good conventions is to named them as `wannier.xyz` and `wannier_x.xyz`, `wannier_y.xyz`, `wannier_z.xyz`.
  * You should use `@include coord_n_cell.inc` to include the coordinate and cell section in the CP2K input files.

A good practice is to create a working directory and put all the files in it. For example, you can create a directory named `al3o2` and put all the files in it. 

Now we can start to build shell script `workflow.sh` to automate the calculation. 

```bash
#!/bin/bash
set -e

# generate slurm template for CP2K
cat << EOF > slurm-cp2k.template
#!/bin/bash
#SBATCH -J cp2k
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -p cpu-large
#SBATCH --mem-per-cpu=4G

module load cp2k/7.1
EOF

# generate CP2K labeling tasks
# note that the tag decide the output file name, 
# for example, --tag dipole's output file will be dipole.out
ai2-kit feat spectr viber cp2k-labeling \
- add_system Al3O2.xyz \
- add_cp2k_input cp2k-dipole.inp --tag dipole \
- add_cp2k_input cp2k-polar-x.inp --tag polar-x \
- add_cp2k_input cp2k-polar-y.inp --tag polar-y \
- add_cp2k_input cp2k-polar-z.inp --tag polar-z \
- make_tasks ./run-01 \
- make_scripts "./run-01/cpk2-batch-{i:02d}.sub" \
--concurrency 5 \
--cp2k_cmd "mpirun cp2k.popt" \
--template slurm-cp2k.template \
- done

# show generated file
find ./run-01 

# submit batch scripts and wait for completion
ai2-kit tool hpc slurm submit ./run-01/cpk2-batch-*.sub - wait

# generate dataset
ai2-kit tool dpdata read run-01/* --fmt cp2k/viber --lumped_dict '{O:4}' --output_file dipole.out \
  --wannier wannier.xyz --wannier_x wannier_x.xyz --wannier_y wannier_y.xyz --wannier_z wannier_z.xyz \
  - write ./al2o3-dataset
```

This script may take a long time to finish, you can run it with `nohup` and use `&` to put it in background. For example

```bash
nohup bash workflow.sh &>> workflow.log &
```

Note that you can view the help document of each command by running with `-h` option, for example 
```bash
ai2-kit feat spectr viber cp2k-labeling - add_system -h
```

After the script finished, you can find the generated dataset in `al2o3-dataset` directory.

### Training Deepmd-kit model



### Predict Dipole and Polarizability
TODO

### Calculate IR and Raman Spectrum
TODO
