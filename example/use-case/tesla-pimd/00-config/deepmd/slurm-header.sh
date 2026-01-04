#!/bin/bash -l
#SBATCH --job-name=dpmd
#SBATCH --time=3:30:00 #HH:MM:SS
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1 #32 MPI ranks per node
#SBATCH --cpus-per-task=72 #8 OMP threads per rank
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive

set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_INTRA_OP_PARALLELISM_THREADS=$SLURM_CPUS_PER_TASK
export TF_INTER_OP_PARALLELISM_THREADS=0

ulimit -s unlimited


