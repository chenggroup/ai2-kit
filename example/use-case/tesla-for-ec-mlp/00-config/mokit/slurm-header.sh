#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --job-name=cp2k
#SBATCH --partition=cpu

set -e
module load miniconda/24.11.1
source activate ec-MLP
export OMP_STACKSIZE=32G
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64

