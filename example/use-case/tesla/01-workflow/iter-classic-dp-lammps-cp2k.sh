#!/bin/bash

set -e

# ensure ITER_NAME is set
[ -z "$ITER_NAME" ] && echo "environment variable ITER_NAME is not set" && exit 1 || echo "ITER_NAME=$ITER_NAME"

CONFIG_DIR=./00-config
WORK_DIR=./20-workdir

# create iter dir
ITER_DIR=$WORK_DIR/iter-$ITER_NAME
mkdir -p $ITER_DIR

[ -f $ITER_DIR/iter.done ] && echo "iteration $ITER_NAME already done" && exit 0 || echo "starting iteration at $ITER_DIR"

# step 1: training

DP_DIR=$ITER_DIR/deepmd
mkdir -p $DP_DIR

[ -f $DP_DIR/setup.done ] && echo "skip deepmd setup" || {
    # generate 4 uniq random seed to train 4 deepmd models
    omb combo \
        add_randint SEED -n 4 -a 0 -b 999999 --uniq - \
        add_var STEPS 5000 - \
        add_file_set DP_DATASET "$WORK_DIR/dp-init-data/*" "$WORK_DIR/iter-*/new-dataset/*" --format json-item --abs - \
        make_files $DP_DIR/model-{i}/input.json --template $CONFIG_DIR/deepmd/input.json - \
        make_files $DP_DIR/model-{i}/run.sh     --template $CONFIG_DIR/deepmd/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$DP_DIR/model-*" - \
        add_header_files $CONFIG_DIR/deepmd/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $DP_DIR/dp-train-{i}.slurm  --concurrency 4

    touch $DP_DIR/setup.done
}

# Submit multiple script to Slurm
omb job slurm submit "$DP_DIR/dp-train*.slurm" --max_tries 2 --wait --recovery $DP_DIR/slurm-recovery.json

# If you are running the workflow on a workstation without Slurm,
# you can just run them as normal shell script, for example:
#
#   parallel -j4 CUDA_VISIBLE_DEVICES='$(({%} - 1))' {} ::: $DP_DIR/dp-train-*.slurm
#
# The above command will make best use of GPUs, for more information, please read
# https://stackoverflow.com/a/79326716/3099733


# step 2: explore
LMP_DIR=$ITER_DIR/lammps
mkdir -p $LMP_DIR

[ -f $LMP_DIR/setup.done ] && echo "skip lammps setup" || {
    omb combo \
        add_files DATA_FILE "$WORK_DIR/lammps-data/*" --abs -\
        add_file_set DP_MODELS "$DP_DIR/model-*/compress.pb" --abs - \
        add_var TEMP 300 500 1000 - \
        add_var STEPS 5000 - \
        add_randint SEED -n 10000 -a 0 -b 99999 - \
        set_broadcast SEED - \
        make_files $LMP_DIR/job-{TEMP}K-{i:03d}/lammps.in --template $CONFIG_DIR/lammps/lammps.in - \
        make_files $LMP_DIR/job-{TEMP}K-{i:03d}/plumed.in --template $CONFIG_DIR/lammps/plumed.in - \
        make_files $LMP_DIR/job-{TEMP}K-{i:03d}/run.sh    --template $CONFIG_DIR/lammps/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$LMP_DIR/job-*" - \
        add_header_files $CONFIG_DIR/lammps/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $LMP_DIR/lammps-{i}.slurm  --concurrency 5

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
        read "$LMP_DIR/job-*/" --traj_file dump.lammpstrj --md_file model_devi.out --specorder "[Ag,O]" --ignore_error - \
        slice "10:" - \
        grade --lo 0.1 --hi 0.2 --col max_devi_f - \
        dump_stats $SCREENING_DIR/stats.tsv - \
        write $SCREENING_DIR/good.xyz   --level good - \
        write $SCREENING_DIR/decent.xyz --level decent - \
        write $SCREENING_DIR/poor.xyz   --level poor - \
        done
    # in the above command slice "10:" is used to skip the first 10 frames of every dump.lammpstrj before grading
    touch $SCREENING_DIR/screening.done
}
cat $SCREENING_DIR/stats.tsv

# step 4: labeling
LABELING_DIR=$ITER_DIR/cp2k
mkdir -p $LABELING_DIR

[ -f $LABELING_DIR/setup.done ] && echo "skip cp2k setup" || {
    # pick 10 frames from the decent.xyz
    ai2-kit tool ase read $SCREENING_DIR/decent.xyz - sample 10  - \
        write_frames $LABELING_DIR/data/{i:03d}.inc --format cp2k-inc

    omb combo \
        add_files DATA_FILE "$LABELING_DIR/data/*" --abs -\
        make_files $LABELING_DIR/job-{i:03d}/cp2k.inp --template $CONFIG_DIR/cp2k/cp2k.inp - \
        make_files $LABELING_DIR/job-{i:03d}/run.sh   --template $CONFIG_DIR/cp2k/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$LABELING_DIR/job-*" - \
        add_header_files $CONFIG_DIR/cp2k/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $LABELING_DIR/cp2k-{i}.slurm  --concurrency 5

    touch $LABELING_DIR/setup.done
}

omb job slurm submit "$LABELING_DIR/cp2k*.slurm" --max_tries 2 --wait --recovery $LABELING_DIR/slurm-recovery.json

# final step: convert cp2k output to dpdata
ai2-kit tool dpdata read $LABELING_DIR/job-*/output --fmt='cp2k/output' --type_map="[Ag,O]" - write $ITER_DIR/new-dataset

# mark iteration as done
touch $ITER_DIR/iter.done
