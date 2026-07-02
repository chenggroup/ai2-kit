#!/bin/bash
set -eu

export CONFIG_DIR=./00-config
export WORK_DIR=./20-workdir
export PROD_DIR=./40-prod
export TYPE_MAP="[Ag,O]"


export DP_MODELS="$WORK_DIR/iter-020/deepmd/model-*/compress.pb"
export MD_STEPS=10000000
export MD_TEMP="200 300 400 500"
export SAMPLE_FREQ=100
export DATA_FILE="$WORK_DIR/lammps-data/000.data"
./01-workflow/prod-dp-lammps.sh
