#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -p cpu
#SBATCH --qos normal
#SBATCH --job-name=cp2k
#SBATCH --mem=249GB  

set -e
module load app/cp2k/2024.1
export OMPI_MCA_btl_openib_allow_ib=1
