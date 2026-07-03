#!/bin/bash

set -e

ITER_NAME="000" TRAINING_STEPS=100000 EXPLORE_STEPS=400 FREQ=2 /public/home/jpqiu/10_interface/mlp/ec-MLP_ai2-kit_example/02-workdir/iter0-cp2k-mokit-dw-ec-lammps.sh
ITER_NAME="001" Labeling_num=40 TRAINING_STEPS=100000 EXPLORE_STEPS=2000 FREQ=10 UPDATE_INITIAL_STR=FALSE /public/home/jpqiu/10_interface/mlp/ec-MLP_ai2-kit_example/02-workdir/iter1-cp2k-mokit-dw-ec-lammps.sh
ITER_NAME="002" Labeling_num=40 TRAINING_STEPS=100000 EXPLORE_STEPS=10000 FREQ=50 UPDATE_INITIAL_STR=FALSE /public/home/jpqiu/10_interface/mlp/ec-MLP_ai2-kit_example/02-workdir/iter1-cp2k-mokit-dw-ec-lammps.sh
ITER_NAME="003" Labeling_num=40 TRAINING_STEPS=100000 EXPLORE_STEPS=20000 FREQ=100 UPDATE_INITIAL_STR=FALSE /public/home/jpqiu/10_interface/mlp/ec-MLP_ai2-kit_example/02-workdir/iter1-cp2k-mokit-dw-ec-lammps.sh
ITER_NAME="004" Labeling_num=40 TRAINING_STEPS=300000 EXPLORE_STEPS=20000 FREQ=100 UPDATE_INITIAL_STR=TRUE /public/home/jpqiu/10_interface/mlp/ec-MLP_ai2-kit_example/02-workdir/iter1-cp2k-mokit-dw-ec-lammps.sh
ITER_NAME="005" Labeling_num=20 TRAINING_STEPS=400000 EXPLORE_STEPS=40000 FREQ=200 UPDATE_INITIAL_STR=TRUE /public/home/jpqiu/10_interface/mlp/ec-MLP_ai2-kit_example/02-workdir/iter1-cp2k-mokit-dw-ec-lammps.sh
#ITER_NAME="003" Labeling_num=40 TRAINING_STEPS=100000 EXPLORE_STEPS=20000 FREQ=100 UPDATE_INITIAL_STR=TRUE /public/home/jpqiu/10_interface/mlp/ec-MLP_ai2-kit_example/02-workdir/iter1-cp2k-mokit-dw-ec-lammps.sh
