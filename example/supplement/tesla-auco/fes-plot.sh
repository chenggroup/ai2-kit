#!/bin/bash
#SBATCH -N 1 -c 8
#SBATCH --job-name=fes-plot
#SBATCH --partition=cpu

set -e

export OMP_NUM_THREADS=32
export TF_INTER_OP_PARALLELISM_THREADS=4
export TF_INTRA_OP_PARALLELISM_THREADS=8

PCMD="/public/groups/ai4ec/libs/conda/deepmd/2.2.6/gpu/bin/plumed sum_hills --bin 99 --min 1.0 --max 5.5 --hills ../HILLS --stride 100"

# Run FES analysis and mult-temperature plot generation on production jobs
python3 fes-plot.py --plumed_cmd "$PCMD" "40-prod/job-*" --out_prefix="prod-"
