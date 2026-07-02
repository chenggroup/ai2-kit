#!/bin/bash
set -eu

module load miniconda
source activate /public/groups/ai4ec/libs/conda/ai2-kit/latest

export CONFIG_DIR=./00-config
export WORK_DIR=./20-workdir
export PROD_DIR=./40-prod
export TYPE_MAP="[Au,C,O]"

export DP_MODELS="$WORK_DIR/iter-100/deepmd/model-*/compress.pb"
export MD_STEPS=10000000
export MD_TEMP="200 300 400 500"
export SAMPLE_FREQ=100
./01-workflow/prod-dp-lammps.sh
