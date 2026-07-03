#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --mem 250G
#SBATCH --job-name=vasp-label
#SBATCH --partition=cpu

set -e
module load vasp/6.4.3-intel