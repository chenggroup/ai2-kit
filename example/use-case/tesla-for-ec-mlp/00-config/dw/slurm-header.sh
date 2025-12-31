#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=lammps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=190G

set -e
module load miniconda/24.11.1
source activate ec-MLP_devel
