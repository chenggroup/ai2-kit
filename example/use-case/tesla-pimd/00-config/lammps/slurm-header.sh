#!/bin/bash -l
#SBATCH --job-name=lammps-pimd
#SBATCH --partition=normal
#SBATCH --time=03:30:00 #HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive

set -e
export OMP_NUM_THREADS=32
export TF_INTRA_OP_PARALLELISM_THREADS=$OMP_NUM_THREADS
export TF_INTER_OP_PARALLELISM_THREADS=0
ulimit -s unlimited

