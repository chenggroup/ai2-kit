#!/bin/bash

set -eu

# ensure the following environment variables are set
omb shell require-env ITER_NAME CONFIG_DIR WORK_DIR TYPE_MAP \
    TRAIN_STEPS DECAY_STEPS MODEL_NUM \
    MD_STEPS MD_TEMP MD_WORKERS SAMPLE_FREQ LAMBDA_f ATOMS_TO_REMOVE \
    MODEL_DEVI_COND USE_BAD_CONFS UPDATE_MD_CONFS \
    LABEL_WORKERS MAX_LABEL \

# constants

ITER_DIR=$WORK_DIR/iter-$ITER_NAME

# create iter dir
mkdir -p $ITER_DIR

[ -f $ITER_DIR/iter.done ] && echo "iteration $ITER_NAME already done" && exit 0
echo "starting iteration at $ITER_DIR"

# step 1: training

DP_DIR=$ITER_DIR/deepmd
mkdir -p $DP_DIR

[ -f $DP_DIR/setup.done ] && echo "skip deepmd setup" || {
    # generate 4 uniq random seed to train 4 deepmd models
    omb combo \
        add_randint SEED -n $MODEL_NUM -a 0 -b 999999 --uniq - \
        add_var TRAIN_STEPS $TRAIN_STEPS - \
        add_var DECAY_STEPS $DECAY_STEPS - \
        add_file_set DP_DATASET "$DATA_DIR/deepmd-init/*" "$WORK_DIR/iter-*/new-dataset-*/*" --format json-item --abs - \
        make_files $DP_DIR/model-{i}/input.json --template $CONFIG_DIR/deepmd/input.json - \
        make_files $DP_DIR/model-{i}/run.sh     --template $CONFIG_DIR/deepmd/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$DP_DIR/model-*" - \
        add_header_files $CONFIG_DIR/deepmd/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $DP_DIR/dp-train-{i}.slurm

    touch $DP_DIR/setup.done
}

# Submit multiple script to Slurm
omb job slurm submit "$DP_DIR/dp-train*.slurm" --max_tries 2 --wait --recovery $DP_DIR/slurm-recovery.json


# step 2: explore
LMP_DIR=$ITER_DIR/lammps
mkdir -p $LMP_DIR

[ -f $LMP_DIR/setup.done ] && echo "skip lammps setup" || {
    omb combo \
        add_files DATA_FILE "$DATA_DIR/lammps-init/h3o/*.data" --abs -\
        add_file_set DP_MODELS "$DP_DIR/model-*/compress.pb" --abs - \
        add_var TEMP $MD_TEMP  - \
        add_var STEPS $MD_STEPS - \
        add_var SAMPLE_FREQ $SAMPLE_FREQ - \
        add_var LAMBDA_f $LAMBDA_f -\
        add_randint SEED -n 10000 -a 0 -b 99999 --uniq - \
        set_broadcast SEED LAMBDA_f - \
        make_files $LMP_DIR/job-{TEMP}K-{LAMBDA_f}-{i:03d}/lammps.in --template $CONFIG_DIR/lammps/lammps.in - \
        make_files $LMP_DIR/job-{TEMP}K-{LAMBDA_f}-{i:03d}/run.sh    --template $CONFIG_DIR/lammps/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$LMP_DIR/job-*" - \
        add_header_files $CONFIG_DIR/lammps/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $LMP_DIR/lammps-{i}.slurm  --concurrency $MD_WORKERS

    touch $LMP_DIR/setup.done
}

omb job slurm submit "$LMP_DIR/lammps*.slurm" --max_tries 2 --wait --recovery $LMP_DIR/slurm-recovery.json

# step 3: screening
SCREENING_DIR=$ITER_DIR/screening
mkdir -p $SCREENING_DIR

[ -f $SCREENING_DIR/screening.done ] && echo "skip screening" || {
    # the ai2-kit model-devi tool is used to screen the candidates
    # use srun if your system has restrictions on the login node
    # for more information, please refer to: https://github.com/chenggroup/ai2-kit/blob/main/doc/manual/model-deviation.md
    ai2-kit tool model_devi \
        read "$LMP_DIR/job-*/" --traj_file dump.lammpstrj --md_file model_devi_ini.out --ignore_error - \
        slice "10:" - \
        grade $MODEL_DEVI_COND --col max_devi_f - \
        dump_stats $SCREENING_DIR/stats-ini.tsv - \
        write $SCREENING_DIR/candidate-ini.xyz --level decent - \
        write $SCREENING_DIR/poor-ini.xyz      --level poor - \
        done

    ai2-kit tool model_devi \
        read "$LMP_DIR/job-*/" --traj_file dump.lammpstrj --md_file model_devi_fin.out --ignore_error - \
        slice "10:" - \
        grade $MODEL_DEVI_COND --col max_devi_f - \
        dump_stats $SCREENING_DIR/stats-fin.tsv - \
        write $SCREENING_DIR/candidate-fin.xyz --level decent - \
        write $SCREENING_DIR/poor-fin.xyz      --level poor - \
        done

    # in the above command slice "10:" is used to skip the first 10 frames of every dump.lammpstrj before grading

    cat $SCREENING_DIR/stats-ini.tsv
    cat $SCREENING_DIR/stats-fin.tsv
    touch $SCREENING_DIR/screening.done
}


# step 4: labeling
LABELING_DIR=$ITER_DIR/cp2k
mkdir -p $LABELING_DIR

[ -f $LABELING_DIR/setup.done ] && echo "skip cp2k setup" || {

    ai2-kit tool ase read $SCREENING_DIR/candidate-ini.xyz - sample $MAX_LABEL  - \
        write_frames $LABELING_DIR/job-ini-{i:03d}/cood_n_cell.inc --format cp2k-inc

    ai2-kit tool ase read $SCREENING_DIR/candidate-fin.xyz - sample $MAX_LABEL  - \
        delete_atoms $ATOMS_TO_REMOVE --start_idx 1 -\
        write_frames $LABELING_DIR/job-fin-{i:03d}/cood_n_cell.inc --format cp2k-inc

    # if USE_BAD_CONFS is greater than 0, we will also add some bad configurations to the labeling set,
    # which can help the model learn better
    [ $USE_BAD_CONFS -gt 0 ] && {
        ai2-kit tool ase read $SCREENING_DIR/poor-ini.xyz - sample $USE_BAD_CONFS --method random - \
            write_frames $LABELING_DIR/job-ini-bad-{i:03d}.inc --format cp2k-inc

        ai2-kit tool ase read $SCREENING_DIR/poor-fin.xyz - sample $USE_BAD_CONFS --method random - \
            delete_atoms $ATOMS_TO_REMOVE --start_idx 1 -\
            write_frames $LABELING_DIR/job-fin-bad-{i:03d}.inc --format cp2k-inc
    }

    # the below use JOR_DIR name pattern to decide the chrage
    omb combo \
        add_files JOR_DIR "$LABELING_DIR/job-*" --abs - \
        compute CHARGE "1 if "job-ini-" in JOB_DIR else 0" - \
        make_files $LABELING_DIR/{JOB_DIR}/cp2k.inp --template $CONFIG_DIR/cp2k/cp2k.inp - \
        make_files $LABELING_DIR/{JOB_DIR}/run.sh   --template $CONFIG_DIR/cp2k/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$LABELING_DIR/job-*" - \
        add_header_files $CONFIG_DIR/cp2k/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $LABELING_DIR/cp2k-{i}.slurm  --concurrency $LABEL_WORKERS

    touch $LABELING_DIR/setup.done
}

omb job slurm submit "$LABELING_DIR/cp2k*.slurm" --max_tries 2 --wait --recovery $LABELING_DIR/slurm-recovery.json

# calculate the error.flag in cp2k model
find $LABELING_DIR/job-*/ -name "error.flag" | wc -l

# final step: convert cp2k output to dpdata
ai2-kit tool dpdata read $LABELING_DIR/job-ini-*/output --fmt='cp2k/output' --type_map="$TYPE_MAP" - write $ITER_DIR/new-dataset-ini
ai2-kit tool dpdata read $LABELING_DIR/job-fin-*/output --fmt='cp2k/output' --type_map="$TYPE_MAP" - write $ITER_DIR/new-dataset-fin

[ $UPDATE_MD_CONFS -gt 0 ] && {
    # replace the initial structures for MD to accelerate the sampling efficiency
    rm -f $DATA_DIR/lammps-init/h3o/* || true
    ai2-kit tool ase read $SCREENING_DIR/candidate-ini.xyz -\
        sample $UPDATE_MD_CONFS --method random - \
        write_frames $WORK_DIR/lammps-init/h3o/{i:03d}.data --format lammps-data --specorder "$TYPE_MAP"
}


# mark iteration as done
touch $ITER_DIR/iter.done
