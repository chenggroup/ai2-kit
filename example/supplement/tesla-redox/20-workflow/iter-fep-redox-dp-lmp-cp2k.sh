#!/bin/bash

set -eu

omb shell require-env ITER_NAME CONFIG_DIR DATA_DIR WORK_DIR TYPE_MAP \
    DEEPMD_INIT_DIR LAMMPS_INIT_DIR \
    TRAIN_STEPS DECAY_STEPS MODEL_NUM \
    MD_STEPS MD_TEMP MD_WORKERS SAMPLE_FREQ LAMBDA_RED \
    MODEL_DEVI_COND LABEL_WORKERS MAX_LABEL

ITER_DIR=$WORK_DIR/iter-$ITER_NAME
mkdir -p $ITER_DIR

[ -f $ITER_DIR/iter.done ] && echo "iteration $ITER_NAME already done" && exit 0
echo "starting iteration at $ITER_DIR"

# step 1: train neutral and reduced DeepMD models in parallel
DP_DIR=$ITER_DIR/deepmd
mkdir -p $DP_DIR

[ -f $DP_DIR/setup-neu.done ] && echo "skip deepmd-neu setup" || {
    omb combo \
        add_randint SEED -n $MODEL_NUM -a 0 -b 999999 --uniq - \
        add_var TRAIN_STEPS $TRAIN_STEPS - \
        add_var DECAY_STEPS $DECAY_STEPS - \
        add_file_set DP_DATASET "$DEEPMD_INIT_DIR/neu-*" "$WORK_DIR/iter-*/new-dataset-neu/*" --format json-item --abs - \
        make_files $DP_DIR/model-neu-{i}/input.json --template $CONFIG_DIR/deepmd/input.json - \
        make_files $DP_DIR/model-neu-{i}/run.sh     --template $CONFIG_DIR/deepmd/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$DP_DIR/model-neu-*" - \
        add_header_files $CONFIG_DIR/deepmd/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $DP_DIR/dp-train-neu-{i}.slurm

    touch $DP_DIR/setup-neu.done
}

[ -f $DP_DIR/setup-red.done ] && echo "skip deepmd-red setup" || {
    omb combo \
        add_randint SEED -n $MODEL_NUM -a 0 -b 999999 --uniq - \
        add_var TRAIN_STEPS $TRAIN_STEPS - \
        add_var DECAY_STEPS $DECAY_STEPS - \
        add_file_set DP_DATASET "$DEEPMD_INIT_DIR/red-*" "$WORK_DIR/iter-*/new-dataset-red/*" --format json-item --abs - \
        make_files $DP_DIR/model-red-{i}/input.json --template $CONFIG_DIR/deepmd/input.json - \
        make_files $DP_DIR/model-red-{i}/run.sh     --template $CONFIG_DIR/deepmd/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$DP_DIR/model-red-*" - \
        add_header_files $CONFIG_DIR/deepmd/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $DP_DIR/dp-train-red-{i}.slurm

    touch $DP_DIR/setup-red.done
}

omb job slurm submit \
    "$DP_DIR/dp-train-neu-*.slurm" \
    "$DP_DIR/dp-train-red-*.slurm" \
    --max_tries 2 --wait \
    --recovery $DP_DIR/slurm-recovery.json

# step 2: explore the alchemical path between neutral and reduced states
LMP_DIR=$ITER_DIR/lammps
mkdir -p $LMP_DIR

[ -f $LMP_DIR/setup.done ] && echo "skip lammps setup" || {
    omb combo \
        add_files DATA_FILE "$LAMMPS_INIT_DIR/*.data" --abs - \
        add_file_set DP_MODELS_NEU "$DP_DIR/model-neu-*/compress.pb" --abs - \
        add_file_set DP_MODELS_RED "$DP_DIR/model-red-*/compress.pb" --abs - \
        add_var TEMP $MD_TEMP - \
        add_var STEPS $MD_STEPS - \
        add_var SAMPLE_FREQ $SAMPLE_FREQ - \
        add_var LAMBDA_RED $LAMBDA_RED - \
        add_randint SEED -n 10000 -a 0 -b 99999 --uniq - \
        set_broadcast SEED LAMBDA_RED - \
        make_files $LMP_DIR/job-{TEMP}K-{LAMBDA_RED}-{i:03d}/lammps.in --template $CONFIG_DIR/lammps/lammps.in - \
        make_files $LMP_DIR/job-{TEMP}K-{LAMBDA_RED}-{i:03d}/run.sh    --template $CONFIG_DIR/lammps/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$LMP_DIR/job-*" - \
        add_header_files $CONFIG_DIR/lammps/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $LMP_DIR/lammps-{i}.slurm --concurrency $MD_WORKERS

    touch $LMP_DIR/setup.done
}

omb job slurm submit "$LMP_DIR/lammps*.slurm" --max_tries 2 --wait --recovery $LMP_DIR/slurm-recovery.json

# step 3: state-specific screening
SCREENING_DIR=$ITER_DIR/screening
mkdir -p $SCREENING_DIR

[ -f $SCREENING_DIR/screening.done ] && echo "skip screening" || {
    ai2-kit tool model_devi \
        read "$LMP_DIR/job-*/" --traj_file dump.lammpstrj --md_file model_devi_neu.out --ignore_error - \
        slice "10:" - \
        grade $MODEL_DEVI_COND --col max_devi_f - \
        dump_stats $SCREENING_DIR/stats-neu.tsv - \
        write $SCREENING_DIR/candidate-neu.xyz --level decent - \
        write $SCREENING_DIR/poor-neu.xyz      --level poor - \
        done

    ai2-kit tool model_devi \
        read "$LMP_DIR/job-*/" --traj_file dump.lammpstrj --md_file model_devi_red.out --ignore_error - \
        slice "10:" - \
        grade $MODEL_DEVI_COND --col max_devi_f - \
        dump_stats $SCREENING_DIR/stats-red.tsv - \
        write $SCREENING_DIR/candidate-red.xyz --level decent - \
        write $SCREENING_DIR/poor-red.xyz      --level poor - \
        done

    cat $SCREENING_DIR/stats-neu.tsv
    cat $SCREENING_DIR/stats-red.tsv
    touch $SCREENING_DIR/screening.done
}

# step 4: label neutral and reduced candidates separately
LABELING_DIR=$ITER_DIR/cp2k
mkdir -p $LABELING_DIR

[ -f $LABELING_DIR/setup.done ] && echo "skip cp2k setup" || {
    ai2-kit tool ase read $SCREENING_DIR/candidate-neu.xyz - sample $MAX_LABEL - \
        write_frames $LABELING_DIR/job-neu-{i:03d}/coord_n_cell.inc --format cp2k-inc

    ai2-kit tool ase read $SCREENING_DIR/candidate-red.xyz - sample $MAX_LABEL - \
        write_frames $LABELING_DIR/job-red-{i:03d}/coord_n_cell.inc --format cp2k-inc

    omb combo \
        add_files JOB_DIR "$LABELING_DIR/job-*" --abs - \
        compute STATE "'neu' if 'job-neu-' in JOB_DIR else 'red'" - \
        make_files {JOB_DIR}/cp2k.inp --template "$CONFIG_DIR/cp2k/cp2k-{STATE}.inp" - \
        make_files {JOB_DIR}/run.sh   --template $CONFIG_DIR/cp2k/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$LABELING_DIR/job-*" - \
        add_header_files $CONFIG_DIR/cp2k/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $LABELING_DIR/cp2k-{i}.slurm --concurrency $LABEL_WORKERS

    touch $LABELING_DIR/setup.done
}

omb job slurm submit "$LABELING_DIR/cp2k*.slurm" --max_tries 2 --wait --recovery $LABELING_DIR/slurm-recovery.json

find $LABELING_DIR/job-*/ -name "error.flag" | wc -l

ai2-kit tool dpdata read $LABELING_DIR/job-neu-*/output --fmt='cp2k/output' --type_map="$TYPE_MAP" --ignore_error - write $ITER_DIR/new-dataset-neu
ai2-kit tool dpdata read $LABELING_DIR/job-red-*/output --fmt='cp2k/output' --type_map="$TYPE_MAP" --ignore_error - write $ITER_DIR/new-dataset-red

touch $ITER_DIR/iter.done
