#!/bin/bash
# OpenMM + MACE exploration job runner (template)
# Variables substituted by omb combo:
#   @STRUCTURE_FILE  - path to starting structure (extxyz)
#   @MACE_MODELS     - space-separated list of all MACE model paths
#   @STEPS           - total MD steps
#   @TEMP            - temperature in Kelvin
#   @SAMPLE_FREQ     - save one frame every N steps
#   @SEED            - random seed
#   @SCRIPT_DIR      - absolute path to 00-config/openmm/
set -e

# Use the first trained model to drive the OpenMM exploration MD
MACE_MODELS_ARR=(@MACE_MODELS)
MD_MODEL="${MACE_MODELS_ARR[0]}"

[ -f openmm.done ] || {
    mambaw run -n openmm python @SCRIPT_DIR/openmm-run.py \
        @STRUCTURE_FILE \
        "$MD_MODEL"     \
        @STEPS          \
        @TEMP           \
        @SAMPLE_FREQ    \
        traj.xyz        \
        @SEED
    touch openmm.done
}

# Compute multi-model deviation and write model_devi.out
[ -f model_devi.done ] || {
    mambaw run -n openmm python @SCRIPT_DIR/model-devi.py traj.xyz model_devi.out @MACE_MODELS
    touch model_devi.done
}
