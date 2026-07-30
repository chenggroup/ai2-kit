#!/bin/bash

set -eu

pip install "ai2-kit>=1.0.8" "oh-my-batch>=0.7.2"

export CONFIG_DIR=./00-config
export DATA_DIR=./10-data
export WORK_DIR=./30-workdir
export TYPE_MAP="[O,H]"
export DEEPMD_INIT_DIR="$DATA_DIR/deepmd-init"
export LAMMPS_INIT_DIR="$DATA_DIR/lammps-init"

export MODEL_NUM=4
export MD_WORKERS=10
export LABEL_WORKERS=20
export MD_TEMP="300 350"
export LAMBDA_RED="0.00 0.25 0.50 0.75 1.00"

export MODEL_DEVI_COND="--lo 0.4 --hi 0.8"
export TRAIN_STEPS=100000
export DECAY_STEPS=1000
export MAX_LABEL=20
export MD_STEPS=1000
export SAMPLE_FREQ=10

ITER_NAME="001" ./20-workflow/iter-fep-redox-dp-lmp-cp2k.sh

export MD_STEPS=4000
export SAMPLE_FREQ=100

ITER_NAME="002" ./20-workflow/iter-fep-redox-dp-lmp-cp2k.sh

export TRAIN_STEPS=400000
export DECAY_STEPS=2000
export MD_STEPS=100000
export SAMPLE_FREQ=100
export MAX_LABEL=50
export MD_TEMP="300 350 400"

ITER_NAME="003" ./20-workflow/iter-fep-redox-dp-lmp-cp2k.sh
ITER_NAME="004" ./20-workflow/iter-fep-redox-dp-lmp-cp2k.sh
ITER_NAME="005" ./20-workflow/iter-fep-redox-dp-lmp-cp2k.sh
