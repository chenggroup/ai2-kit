#!/bin/bash

set -eu

pip install "ai2-kit>=1.0.8" "oh-my-batch>=0.7.2"

export CONFIG_DIR=./00-config
export WORK_DIR=./20-workdir
export TYPE_MAP="[O,H]"

export PROTON_TO_REOMVE=4 # index of lammps atom id (1-based)
export MODEL_NUM=4
export MD_WORKERS=10
export LABEL_WORKERS=20
export MD_TEMP="330 430 530"
export LAMBDA_f="0.00 0.25 0.50 0.75 1.00"
export MODEL_DEVI_COND="--lo 0.2 --hi 0.4"
export TRAIN_STEPS=100000
export DECAY_STEPS=1000
export MAX_LABEL=20
export USE_BAD_CONFS=0
export UPDATE_MD_CONFS=0
export MD_STEPS=1000
export SAMPLE_FREQ=10

ITER_NAME="001" ./01-workflow/iter-fep-pka-dp-lammps-cp2k.sh

export MD_STEPS=4000
export SAMPLE_FREQ=100

ITER_NAME="002" ./01-workflow/iter-fep-pka-dp-lammps-cp2k.sh

export TRAIN_STEPS=400000
export DECAY_STEPS=2000
export MD_STEPS=100000
export SAMPLE_FREQ=100
export UPDATE_MD_CONFS=2
export MAX_LABEL=50
export MD_TEMP="330 430 530 630"

ITER_NAME="003" ./01-workflow/iter-fep-pka-dp-lammps-cp2k.sh
ITER_NAME="004" ./01-workflow/iter-fep-pka-dp-lammps-cp2k.sh
ITER_NAME="005" ./01-workflow/iter-fep-pka-dp-lammps-cp2k.sh


