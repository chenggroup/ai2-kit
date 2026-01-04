#!/bin/bash

set -e

# ensure ITER_NAME is set
[ -z "$ITER_NAME" ] && echo "environment variable ITER_NAME is not set" && exit 1 || echo "ITER_NAME=$ITER_NAME"

CONFIG_DIR=./00-config
WORK_DIR=./02-workdir

# create iter dir
ITER_DIR=$WORK_DIR/iter-$ITER_NAME
mkdir -p $ITER_DIR

[ -f $ITER_DIR/iter.done ] && echo "iteration $ITER_NAME already done" && exit 0 || echo "starting iteration at $ITER_DIR"

# step 1: training

DP_DIR=$ITER_DIR/deepmd
mkdir -p $DP_DIR

[ -f $DP_DIR/setup.done ] && echo "skip deepmd setup" || {
    omb combo \
        add_seq MODEL_ID 0 4 - \
        add_var STEPS 400000 - \
        add_randint SEED -n 4 -a 100000 -b 999999 --uniq - \
        add_file_set DP_DATASET "$WORK_DIR/dp-init-data/*" "$WORK_DIR/iter-*/new-dataset/*" --format json-item --abs - \
        set_broadcast SEED - \
        make_files $DP_DIR/model-{MODEL_ID}/input.json --template $CONFIG_DIR/deepmd/input.json - \
        make_files $DP_DIR/model-{MODEL_ID}/run.sh     --template $CONFIG_DIR/deepmd/run.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$DP_DIR/model-*" - \
        add_header_files $CONFIG_DIR/deepmd/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $DP_DIR/dp-train-{i}.slurm  --concurrency 4
    touch $DP_DIR/setup.done
}

omb job slurm submit "$DP_DIR/dp-train*.slurm" --max_tries 2 --wait --recovery $DP_DIR/slurm-recovery.json

# step 2: explore
LMP_DIR=$ITER_DIR/lammps
mkdir -p $LMP_DIR

[ -f $LMP_DIR/setup.done ] && echo "skip lammps setup" || {
    omb combo \
        add_files DATA_FILE "$WORK_DIR/xyz-data/*" --abs - \
        add_file_set DP_MODELS "$DP_DIR/model-*/compress.pb" --abs - \
        add_var TEMP 350 - \
        add_var STEPS 100000 - \
        add_randint SEED -n 10000 -a 0 -b 99999 - \
        set_broadcast SEED - \
        make_files $LMP_DIR/job-{TEMP}K-{i:03d}/lammps.in  --template $CONFIG_DIR/lammps/lammps.in - \
        make_files $LMP_DIR/job-{TEMP}K-{i:03d}/plumed.inp --template $CONFIG_DIR/lammps/plumed.inp - \
        make_files $LMP_DIR/job-{TEMP}K-{i:03d}/input.xml  --template $CONFIG_DIR/lammps/input.xml - \
        make_files $LMP_DIR/job-{TEMP}K-{i:03d}/conf.lmp   --template $CONFIG_DIR/lammps/conf.lmp - \
        make_files $LMP_DIR/job-{TEMP}K-{i:03d}/run.sh     --template $CONFIG_DIR/lammps/run.sh --mode 755 - \
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
    # for more information, please refer to: https://github.com/chenggroup/ai2-kit/blob/main/doc/manual/model-deviation.md
    # Taking structures from one client would be sufficient.
    ai2-kit tool model_devi \
	read "$LMP_DIR/job-*/" --traj_file dump-0.lammpstrj --specorder "[Bi,H,O,V]" --md_file model_devi-0.out --ignore_error - \
        read "$LMP_DIR/job-*/" --traj_file dump-1.lammpstrj --specorder "[Bi,H,O,V]" --md_file model_devi-1.out --ignore_error - \
        read "$LMP_DIR/job-*/" --traj_file dump-2.lammpstrj --specorder "[Bi,H,O,V]" --md_file model_devi-2.out --ignore_error - \
        read "$LMP_DIR/job-*/" --traj_file dump-3.lammpstrj --specorder "[Bi,H,O,V]" --md_file model_devi-3.out --ignore_error - \
        slice "10:" - \
        grade --lo 0.2 --hi 0.7 --col max_devi_f - \
        dump_stats $SCREENING_DIR/stats.tsv - \
        write $SCREENING_DIR/decent.xyz --level decent - \
        done
    # in the above command slice "10:" is used to skip the first 10 frames in each dump.lammpstrj
    touch $SCREENING_DIR/screening.done
}
cat $SCREENING_DIR/stats.tsv


# step 4: labeling
LABELING_DIR=$ITER_DIR/cp2k
mkdir -p $LABELING_DIR

[ -f $LABELING_DIR/setup.done ] && echo "skip cp2k setup" || {
    # drop the first 10 frames and then randomly pick 50 frmaes to label
    ai2-kit tool ase read $SCREENING_DIR/decent.xyz --index 10:: - sample 50 --method random - \
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
        make $LABELING_DIR/cp2k-{i}.slurm --concurrency 25

    touch $LABELING_DIR/setup.done
}

omb job slurm submit "$LABELING_DIR/cp2k*.slurm" --max_tries 3 --wait --recovery $LABELING_DIR/slurm-recovery.json

# final step: convert cp2k output to dpdata
ai2-kit tool dpdata read $LABELING_DIR/job-*/output --fmt='cp2k/output' --type_map="[Bi,H,O,V]" - write $ITER_DIR/new-dataset

# mark iteration as done
touch $ITER_DIR/iter.done
