#! /bin/bash

module load anaconda/2022.5
source ~/.bashrc
conda activate ai2-kit

# omb command
sh ./omb/omb.sub

# train models command
sh ./train/train.sub

# run molecular dynamics command
sh ./simulate/lammps.sub

# predict dipole moments and polarizabilities command
sh ./predict/predict.sub

# calculate spectra command
sh ./spectra/spectra.sub