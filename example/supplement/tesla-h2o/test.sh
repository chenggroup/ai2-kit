#!/bin/bash

#SBATCH -N 1 -c 32
#SBATCH --job-name=dptest
#SBATCH --partition=cpu

set -e
module load anaconda/2022.5
module load cuda/11.6
source activate /public/groups/ai4ec/libs/conda/deepmd/2.2.6/gpu

export OMP_NUM_THREADS=32
export TF_INTER_OP_PARALLELISM_THREADS=4
export TF_INTRA_OP_PARALLELISM_THREADS=8

MODEL_PATH=./20-workdir/iter-005/deepmd/model-0/compress.pb

ls -1d ./20-workdir/iter-00?/new-dataset/* \
       ./20-workdir/dp-init-data/* > datafile.txt

rm -rf ./30-dp-test || true
mkdir -p ./30-dp-test
dp test -m $MODEL_PATH -f datafile.txt --detail-file ./30-dp-test/test
rm datafile.txt
