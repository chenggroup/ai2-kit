#!/bin/bash

set -eu

module load miniconda
source activate /public/groups/ai4ec/libs/conda/ai2-kit/latest

export CONFIG_DIR=./00-config
export WORK_DIR=./20-workdir
export TYPE_MAP="[Au,C,O]"

./01-workflow/setup.sh

export MODEL_NUM=4
export MD_WORKERS=20
export LABEL_WORKERS=100
export USE_BAD_CONFS=0

export MD_TEMP="100 200 300 400 600 800 1000"
export MODEL_DEVI_COND="--lo 0.15 --hi 0.45 --outlier 0.8"

export TRAIN_STEPS=200000
export DECAY_STEPS=2000
export SAMPLE_FREQ=100

export MAX_LABEL=100
export MD_STEPS=30000

ITER_NAME="001" ./01-workflow/iter-classic-dp-lammps-cp2k.sh
ITER_NAME="002" ./01-workflow/iter-classic-dp-lammps-cp2k.sh
ITER_NAME="003" ./01-workflow/iter-classic-dp-lammps-cp2k.sh
ITER_NAME="004" ./01-workflow/iter-classic-dp-lammps-cp2k.sh
ITER_NAME="005" ./01-workflow/iter-classic-dp-lammps-cp2k.sh

export MAX_LABEL=200
export MD_STEPS=100000
ITER_NAME="006" ./01-workflow/iter-classic-dp-lammps-cp2k.sh
ITER_NAME="007" ./01-workflow/iter-classic-dp-lammps-cp2k.sh
ITER_NAME="008" ./01-workflow/iter-classic-dp-lammps-cp2k.sh
ITER_NAME="009" ./01-workflow/iter-classic-dp-lammps-cp2k.sh
ITER_NAME="010" ./01-workflow/iter-classic-dp-lammps-cp2k.sh

# Final train and test
export MAX_LABEL=0
export TRAIN_STEPS=2000000
export DECAY_STEPS=20000
export MD_STEPS=300000
ITER_NAME="100" ./01-workflow/iter-classic-dp-lammps-cp2k.sh
