#!/bin/bash
# MACE model training script (template)
# Variables substituted by omb combo: @SEED, @TRAIN_EPOCHS, @TRAIN_FILE
set -e

[ -f train.done ] || {

    mambaw run -n openmm mace_run_train \
        --name="mace_model" \
        --train_file="../all.xyz" \
        --valid_fraction=0.1 \
        --E0s="average" \
        --model="MACE" \
        --num_interactions=2 \
        --num_channels=128 \
        --energy_key=energy \
        --forces_key=forces \
        --max_L=1 \
        --correlation=3 \
        --r_max=5.0 \
        --batch_size=@BATCH_SIZE \
        --patience=@PATIENT_EPOCHS \
        --max_num_epochs=@TRAIN_EPOCHS \
        --seed=@SEED \
        --device=cuda \
        --forces_weight=10.0 \
        --energy_weight=1.0 \
        --swa \
        --swa_energy_weight=1.0 \
        --swa_forces_weight=1.0 \
        --lr=0.001 --swa_lr=0.0001 \
        --save_cpu
    touch train.done
}
