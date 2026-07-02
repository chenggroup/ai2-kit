#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --job-name=cp2k
#SBATCH --partition=cpu

set -e
module load intel/oneapi2021.1
module load cp2k/7.1