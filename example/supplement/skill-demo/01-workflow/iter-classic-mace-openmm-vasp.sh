#!/bin/bash
# TESLA iteration: MACE (training) → OpenMM+MACE (exploration) → model_devi (screening) → VASP (labeling)
set -eu

# ensure required environment variables are set
omb shell require-env ITER_NAME CONFIG_DIR WORK_DIR TYPE_MAP \
    TRAIN_EPOCHS MODEL_NUM BATCH_SIZE PATIENT_EPOCHS \
    MD_STEPS MD_TEMP MD_WORKERS SAMPLE_FREQ \
    MODEL_DEVI_COND USE_BAD_CONFS UPDATE_MD_CONFS \
    LABEL_WORKERS MAX_LABEL

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ITER_DIR=$WORK_DIR/iter-$ITER_NAME

mkdir -p $ITER_DIR

[ -f $ITER_DIR/iter.done ] && echo "iteration $ITER_NAME already done" && exit 0
echo "starting iteration at $ITER_DIR"

# ===========================================================================
# Step 1: Training — MACE
# ===========================================================================
MACE_DIR=$ITER_DIR/mace
mkdir -p $MACE_DIR

[ -f $MACE_DIR/setup.done ] && echo "skip mace setup" || {

    # merge initial dataset and newly labeled dataset from last iteration (if exists)
    cat $WORK_DIR/mace-init-data/*.xyz \
        $WORK_DIR/iter-*/new-dataset/*.xyz > $MACE_DIR/all.xyz || true

    # Generate one training directory per model with a unique random seed.
    omb combo \
        add_randint SEED -n $MODEL_NUM -a 0 -b 999999 --uniq - \
        add_var BATCH_SIZE $BATCH_SIZE - \
        add_var PATIENT_EPOCHS $PATIENT_EPOCHS - \
        add_var TRAIN_EPOCHS $TRAIN_EPOCHS - \
        make_files $MACE_DIR/model-{i}/run.sh --template $CONFIG_DIR/mace/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$MACE_DIR/model-*" - \
        add_header_files $CONFIG_DIR/mace/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $MACE_DIR/mace-train-{i}.slurm

    touch $MACE_DIR/setup.done
}

omb job slurm submit "$MACE_DIR/mace-train*.slurm" --max_tries 2 --wait --recovery $MACE_DIR/slurm-recovery.json

# ===========================================================================
# Step 2: Exploration — OpenMM driven by a single MACE model
#          Model deviation is computed post-hoc across all trained models.
# ===========================================================================
OPENMM_DIR=$ITER_DIR/openmm
mkdir -p $OPENMM_DIR

[ -f $OPENMM_DIR/setup.done ] && echo "skip openmm setup" || {
    omb combo \
        add_files     STRUCTURE_FILE "$WORK_DIR/openmm-data/*" --abs - \
        add_file_set  MACE_MODELS    "$MACE_DIR/model-*/mace_model_stagetwo.model" --abs - \
        add_var       TEMP           $MD_TEMP - \
        add_var       STEPS          $MD_STEPS - \
        add_var       SAMPLE_FREQ    $SAMPLE_FREQ - \
        add_var       SCRIPT_DIR     "$(realpath $CONFIG_DIR/openmm)" - \
        add_randint   SEED -n 10000 -a 0 -b 99999 --uniq - \
        set_broadcast SEED - \
        make_files $OPENMM_DIR/job-{TEMP}K-{i:03d}/run.sh --template $CONFIG_DIR/openmm/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$OPENMM_DIR/job-*" - \
        add_header_files $CONFIG_DIR/openmm/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $OPENMM_DIR/openmm-{i}.slurm --concurrency $MD_WORKERS

    touch $OPENMM_DIR/setup.done
}

omb job slurm submit "$OPENMM_DIR/openmm*.slurm" --max_tries 2 --wait --recovery $OPENMM_DIR/slurm-recovery.json

# ===========================================================================
# Step 3: Screening — grade frames by multi-model force deviation
# ===========================================================================
SCREENING_DIR=$ITER_DIR/screening
mkdir -p $SCREENING_DIR

[ -f $SCREENING_DIR/screening.done ] && echo "skip screening" || {
    # traj_file is traj.xyz (extxyz) written by openmm-run.py
    # slice "10:" skips the first 10 frames (equilibration)
    ai2-kit tool model_devi \
        read "$OPENMM_DIR/job-*/" \
            --traj_file traj.xyz \
            --md_file   model_devi.out \
            --format    extxyz \
            --ignore_error - \
        slice "10:" - \
        grade $MODEL_DEVI_COND --col max_devi_f - \
        dump_stats $SCREENING_DIR/stats.tsv - \
        write $SCREENING_DIR/good.xyz   --level good - \
        write $SCREENING_DIR/decent.xyz --level decent - \
        write $SCREENING_DIR/poor.xyz   --level poor - \
        done

    touch $SCREENING_DIR/screening.done
}
cat $SCREENING_DIR/stats.tsv

# Exit early when no candidate structures were found
if [ ! -s $SCREENING_DIR/decent.xyz ]; then
    echo "no decent structure found; iteration considered done"
    touch $ITER_DIR/iter.done
    exit 1
fi

# ===========================================================================
# Step 4: Labeling — VASP single-point energy and forces
# ===========================================================================
LABELING_DIR=$ITER_DIR/vasp
mkdir -p $LABELING_DIR

[ -f $LABELING_DIR/setup.done ] && echo "skip vasp setup" || {
    # Write each candidate frame as a separate POSCAR file
    ai2-kit tool ase read $SCREENING_DIR/decent.xyz --format extxyz - \
        sample $MAX_LABEL - \
        write_frames $LABELING_DIR/data/{i:03d}.vasp --format vasp

    # Also label a small number of "poor" (high-deviation) structures to help
    # the model learn better if USE_BAD_CONFS > 0
    [ $USE_BAD_CONFS -gt 0 ] && {
        ai2-kit tool ase read $SCREENING_DIR/poor.xyz --format extxyz - \
            sample $USE_BAD_CONFS --method random - \
            write_frames $LABELING_DIR/data/bad-{i:03d}.vasp --format vasp
    }

    omb combo \
        add_files POSCAR_FILE   "$LABELING_DIR/data/*" --abs - \
        add_files INCAR_FILE    "$CONFIG_DIR/vasp/INCAR"    --abs - \
        add_files KPOINTS_FILE  "$CONFIG_DIR/vasp/KPOINTS"  --abs - \
        add_files POTCAR_FILE   "$CONFIG_DIR/vasp/POTCAR"   --abs - \
        make_files $LABELING_DIR/job-{i:03d}/run.sh --template $CONFIG_DIR/vasp/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$LABELING_DIR/job-*" - \
        add_header_files $CONFIG_DIR/vasp/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $LABELING_DIR/vasp-{i}.slurm --concurrency $LABEL_WORKERS

    touch $LABELING_DIR/setup.done
}

omb job slurm submit "$LABELING_DIR/vasp*.slurm" --max_tries 2 --wait --recovery $LABELING_DIR/slurm-recovery.json

# Report failed VASP jobs (non-zero count is not fatal; bad structures are skipped)
echo "VASP jobs with error.flag: $(find $LABELING_DIR/job-*/ -name 'error.flag' | wc -l)"

# ===========================================================================
# Final step: Convert VASP OUTCAR → extxyz and accumulate as training data
# ===========================================================================
# ASE reads energies and forces from OUTCAR; --ignore_error skips failed jobs
ai2-kit tool ase \
    read "$LABELING_DIR/job-*/OUTCAR" --format vasp-out --ignore_error - \
    write $ITER_DIR/new-dataset/dataset.xyz --format extxyz

# Optionally refresh the OpenMM starting structures from well-converged frames
# to accelerate sampling efficiency in the next iteration.
[ $UPDATE_MD_CONFS -gt 0 ] && {
    rm -f $WORK_DIR/openmm-data/* || true
    ai2-kit tool ase read $SCREENING_DIR/good.xyz --format extxyz - \
        sample $UPDATE_MD_CONFS --method random - \
        write_frames $WORK_DIR/openmm-data/{i:03d}.xyz --format extxyz
}

touch $ITER_DIR/iter.done
