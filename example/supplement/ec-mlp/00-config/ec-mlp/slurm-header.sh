#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=ec-mlp
##SBATCH --partition=gpu
#SBATCH --partition=gpu,gpu-mig-2g-20gb
#SBATCH --gres=gpu:1
#SBATCH --mem=190G

set -e
module load miniconda/24.11.1
source activate /public/groups/chenggroup/jpqiu/conda/ec-MLP
export DP_INFER_BATCH_SIZE=32768 
