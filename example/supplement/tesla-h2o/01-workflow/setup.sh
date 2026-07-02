#!/bin/bash

# this script is to generate deepmd train data and lammps structure files from aimd.xyz
set -e

omb shell require-env TYPE_MAP WORK_DIR CONFIG_DIR


[ -f $WORK_DIR/setup.done ] && echo "setup already done" && exit 0
echo "starting setup at $WORK_DIR"

# select 50 frames from aimd trajectory as deepmd init dataset
ai2-kit tool ase read ./00-config/aimd.xyz - sample 50 - to_dpdata --labeled --type_map "$TYPE_MAP" - write $WORK_DIR/dp-init-data
# select 2 frames from aimd trajectory as lammps init dataset
ai2-kit tool ase read ./00-config/aimd.xyz - sample 2 - write_frames $WORK_DIR/lammps-data/{i:03d}.data --format lammps-data --specorder "$TYPE_MAP"

touch $WORK_DIR/setup.done
