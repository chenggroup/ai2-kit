#!/bin/bash

#SBATCH -N 2
#SBATCH --ntasks-per-node=64
#SBATCH --job-name=cp2k
#SBATCH --partition=cpu

set -e
module load cp2k/9.1
