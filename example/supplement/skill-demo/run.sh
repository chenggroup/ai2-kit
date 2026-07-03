#!/bin/bash

set -eu

pip install "ai2-kit>=1.0.9" "oh-my-batch>=0.7.2"

export CONFIG_DIR=./00-config
export WORK_DIR=./20-workdir
export TYPE_MAP="[O,H]"

# variables needed by setup.sh
export LABEL_WORKERS=20
export INIT_LABEL=50    # number of aimd.xyz frames to relabel with VASP for initial training data

./01-workflow/setup.sh

export MODEL_NUM=4
export MD_WORKERS=10
export MD_TEMP="330 430 530"
export MODEL_DEVI_COND="--lo 0.4 --hi 1.0"
export MAX_LABEL=20
export USE_BAD_CONFS=0
export UPDATE_MD_CONFS=0
export BATCH_SIZE=16

export TRAIN_EPOCHS=400
export PATIENT_EPOCHS=50
export MD_STEPS=1000
export SAMPLE_FREQ=10

ITER_NAME="001" ./01-workflow/iter-classic-mace-openmm-vasp.sh

export MD_STEPS=4000
export SAMPLE_FREQ=100

ITER_NAME="002" ./01-workflow/iter-classic-mace-openmm-vasp.sh

export TRAIN_EPOCHS=600
export PATIENT_EPOCHS=60
export MD_STEPS=100000
export SAMPLE_FREQ=100
export UPDATE_MD_CONFS=2
export MAX_LABEL=50

ITER_NAME="003" ./01-workflow/iter-classic-mace-openmm-vasp.sh
ITER_NAME="004" ./01-workflow/iter-classic-mace-openmm-vasp.sh
ITER_NAME="005" ./01-workflow/iter-classic-mace-openmm-vasp.sh


