#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=mace-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e
module load singularity
