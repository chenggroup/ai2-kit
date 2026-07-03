#!/bin/bash

# Generate initial MACE training data by relabeling aimd.xyz frames with VASP,
# and prepare starting structures for OpenMM exploration.
set -e

omb shell require-env TYPE_MAP WORK_DIR CONFIG_DIR LABEL_WORKERS INIT_LABEL


[ -f $WORK_DIR/setup.done ] && echo "setup already done" && exit 0
echo "starting setup at $WORK_DIR"

INIT_LABEL_DIR=$WORK_DIR/init-labeling
mkdir -p $INIT_LABEL_DIR

# ---------------------------------------------------------------------------
# Stage 1: VASP relabeling of sampled aimd.xyz frames
# The aimd.xyz was produced by CP2K, so energies/forces are not compatible
# with the VASP functional we use here.  We must recompute them with VASP.
# ---------------------------------------------------------------------------
[ -f $INIT_LABEL_DIR/setup.done ] && echo "skip init-labeling setup" || {
    # Write each sampled frame as a POSCAR file
    ai2-kit tool ase read ./00-config/aimd.xyz --format extxyz - \
        sample $INIT_LABEL - \
        write_frames $INIT_LABEL_DIR/data/{i:03d}.vasp --format vasp

    omb combo \
        add_files POSCAR_FILE  "$INIT_LABEL_DIR/data/*"      --abs - \
        add_files INCAR_FILE   "$CONFIG_DIR/vasp/INCAR"       --abs - \
        add_files KPOINTS_FILE "$CONFIG_DIR/vasp/KPOINTS"     --abs - \
        add_files POTCAR_FILE  "$CONFIG_DIR/vasp/POTCAR"      --abs - \
        make_files $INIT_LABEL_DIR/job-{i:03d}/run.sh --template $CONFIG_DIR/vasp/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$INIT_LABEL_DIR/job-*" - \
        add_header_files $CONFIG_DIR/vasp/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $INIT_LABEL_DIR/vasp-{i}.slurm --concurrency $LABEL_WORKERS

    touch $INIT_LABEL_DIR/setup.done
}

omb job slurm submit "$INIT_LABEL_DIR/vasp*.slurm" --max_tries 2 --wait --recovery $INIT_LABEL_DIR/slurm-recovery.json

echo "VASP init-labeling jobs with error.flag: $(find $INIT_LABEL_DIR/job-*/ -name 'error.flag' | wc -l)"

# Convert VASP OUTCARs to a single extxyz file used as the MACE initial dataset
ai2-kit tool ase \
    read "$INIT_LABEL_DIR/job-*/OUTCAR" --format vasp-out --ignore_error - \
    write $WORK_DIR/mace-init-data/dataset.xyz --format extxyz

# ---------------------------------------------------------------------------
# Stage 2: Sample starting structures for OpenMM exploration (positions only)
# ---------------------------------------------------------------------------
ai2-kit tool ase read ./00-config/aimd.xyz --format extxyz - sample 2 - \
    write_frames $WORK_DIR/openmm-data/{i:03d}.xyz --format extxyz

touch $WORK_DIR/setup.done
