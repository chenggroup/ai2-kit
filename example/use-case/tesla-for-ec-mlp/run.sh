#!/bin/bash

set -e

ITER_NAME="000" TRAINING_STEPS=100000 EXPLORE_STEPS=100 FREQ=1 /public/home/jpqiu/10_interface/mlp/1_Cu100/0_cl_cover/02-workdir/iter0-cp2k-mokit-dw-ec-lammps.sh
ITER_NAME="001" Labeling_num=40 TRAINING_STEPS=100000 EXPLORE_STEPS=800 FREQ=4 UPDATE_INITIAL_STR=FALSE /public/home/jpqiu/10_interface/mlp/1_Cu100/0_cl_cover/02-workdir/iter1-cp2k-mokit-dw-ec-lammps.sh
ITER_NAME="002" Labeling_num=60 TRAINING_STEPS=100000 EXPLORE_STEPS=1200 FREQ=6 UPDATE_INITIAL_STR=FALSE /public/home/jpqiu/10_interface/mlp/1_Cu100/0_cl_cover/02-workdir/iter1-cp2k-mokit-dw-ec-lammps.sh
ITER_NAME="003" Labeling_num=60 TRAINING_STEPS=100000 EXPLORE_STEPS=2000 FREQ=10 UPDATE_INITIAL_STR=FALSE /public/home/jpqiu/10_interface/mlp/1_Cu100/0_cl_cover/02-workdir/iter1-cp2k-mokit-dw-ec-lammps.sh
#ITER_NAME="002" Labeling_num=80 TRAINING_STEPS=100000 EXPLORE_STEPS=20000 FREQ=100 UPDATE_INITIAL_STR=FALSE /public/home/jpqiu/10_interface/mlp/1_Cu100/0_cl_cover/02-workdir/iter1-cp2k-mokit-dw-ec-lammps.sh
#ITER_NAME="003" Labeling_num=80 TRAINING_STEPS=100000 EXPLORE_STEPS=80000 FREQ=400 UPDATE_INITIAL_STR=FALSE /public/home/jpqiu/10_interface/mlp/1_Cu100/0_cl_cover/02-workdir/iter2-cp2k-mokit-dw-ec-lammps.sh
