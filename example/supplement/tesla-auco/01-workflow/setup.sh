#!/bin/bash

# this script is to generate deepmd train data and lammps structure files from aimd.xyz
set -eu

omb shell require-env TYPE_MAP WORK_DIR CONFIG_DIR

[ -f $WORK_DIR/setup.done ] && echo "setup already done" && exit 0
echo "starting setup at $WORK_DIR"

# select 20 frames from aimd trajectory as deepmd init dataset
ai2-kit tool ase read ./00-config/aimd.xyz - sample 100 - to_dpdata --labeled - write $WORK_DIR/dp-init-data

touch $WORK_DIR/setup.done
