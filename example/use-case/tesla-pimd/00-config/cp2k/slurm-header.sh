#!/bin/bash -l
#SBATCH --job-name=cp2k_oer
#SBATCH --time=04:00:00 #HH:MM:SS
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=64 #32 MPI ranks per node
#SBATCH --cpus-per-task=4 #8 OMP threads per rank
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive
#SBATCH --uenv=cp2k/2024.3:v2
#SBATCH --view=cp2k


set -e
export CP2K_DATA_DIR="/capstor/store/cscs/userlab/lp07/zyongbin/software/cp2k/data"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_CACHE_PATH="/dev/shm/$RANDOM"
export MPICH_MALLOC_FALLBACK=1
export MPICH_GPU_SUPPORT_ENABLED=1


