#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=lammps
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1

set -e
module load intel/2023.2
module load gsl/2.8
module load miniconda/24.11.1
source activate ec-MLP_devel 
