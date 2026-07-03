#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=openmm-mace
#SBATCH --partition=gpu-mig-2g-20gb,gpu
#SBATCH --gres=gpu:1

set -e
module load singularity
