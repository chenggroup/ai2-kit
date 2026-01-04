#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=lammps
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e
module load anaconda/2022.5
module load cuda/11.6
source activate /public/groups/ai4ec/libs/conda/deepmd/2.2.6/gpu
export OMP_NUM_THREADS=8
export TF_INTER_OP_PARALLELISM_THREADS=2
export TF_INTRA_OP_PARALLELISM_THREADS=4