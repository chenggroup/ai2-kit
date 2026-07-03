#!/bin/bash

set -e

check_env() {
    local var=$1
    if [ -z "${!var}" ]; then
        echo "❌ Environment variable $var is not set"
        exit 1
    else
        echo "✅ $var=${!var}"
    fi
}
check_env ITER_NAME
check_env TRAINING_STEPS
check_env EXPLORE_STEPS
check_env FREQ

# initial computational setup
CONFIG_DIR=/public/home/jpqiu/10_interface/mlp/ec-MLP_ai2-kit_example/00-config
WORK_DIR=/public/home/jpqiu/10_interface/mlp/ec-MLP_ai2-kit_example/02-workdir
INIT_DATASET_DIR=/public/home/jpqiu/10_interface/mlp/ec-MLP_ai2-kit_example/01-init/00-data/
SYS_EXPLORE_DIR=/public/home/jpqiu/10_interface/mlp/ec-MLP_ai2-kit_example/01-init/01-explore/ 
SYS_NAMES="water"
declare -A CHARGE_MAP=(
    [water]=0
)

# create iter dir
ITER_DIR=$WORK_DIR/iter-$ITER_NAME
mkdir -p $ITER_DIR

[ -f $ITER_DIR/iter.done ] && echo "iteration $ITER_NAME already done" && exit 0 || echo "starting iteration at $ITER_DIR"

PREV_ITER_NAME=$(printf "%03d" $((10#$ITER_NAME - 1)))
PREV_ITER_DIR=$WORK_DIR/iter-$PREV_ITER_NAME

# step 3: training dw model

DW_DIR=$ITER_DIR/dw
mkdir -p $DW_DIR
DECAY=$((TRAINING_STEPS / 200))

[ -f $DW_DIR/dw-setup.done ] && echo "skip dw setup" || {
    # generate 1 random seed to train 1 dw models
    omb combo \
        add_randint SEED -n 1 -a 0 -b 999999 --uniq - \
        add_var STEPS $TRAINING_STEPS - \
        add_var DECAY $DECAY - \
        add_file_set DP_DATASET \
             "$INIT_DATASET_DIR/*" \
             --format json-item --abs - \
        make_files $DW_DIR/dw-model-{i}/dw.json --template $CONFIG_DIR/dw/dw.json - \
        make_files $DW_DIR/dw-model-{i}/dw.sh     --template $CONFIG_DIR/dw/dw.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$DW_DIR/dw-model-*" - \
        add_header_files $CONFIG_DIR/dw/slurm-header.sh - \
        add_cmds "bash ./dw.sh" - \
        make $DW_DIR/dw-train-{i}.slurm  --concurrency 1

    touch $DW_DIR/dw-setup.done
}
cd $DW_DIR
omb job slurm submit "$DW_DIR/dw-train*.slurm" --max_tries 2 --wait --recovery $DW_DIR/slurm-recovery.json

# step 4: training ec-mlp model

EC_DIR=$ITER_DIR/ec-mlp
mkdir -p $EC_DIR

[ -f $EC_DIR/ec-mlp-setup.done ] && echo "skip ec-mlp setup" || {
    omb combo \
        add_randint SEED -n 4 -a 0 -b 999999 --uniq - \
        add_var STEPS $TRAINING_STEPS - \
        add_var DECAY $DECAY - \
        add_file_set DW_MODEL  "$DW_DIR/dw-model-*/dw.pb" --format json-item --abs - \
        add_file_set DP_DATASET  \
             "$INIT_DATASET_DIR/*" \
             --format json-item --abs - \
        make_files $EC_DIR/ec-mlp-model-{i}/ec-mlp.json --template $CONFIG_DIR/ec-mlp/ec-mlp.json - \
        make_files $EC_DIR/ec-mlp-model-{i}/ec-mlp.sh     --template $CONFIG_DIR/ec-mlp/ec-mlp.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$EC_DIR/ec-mlp-model-*" - \
        add_header_files $CONFIG_DIR/ec-mlp/slurm-header.sh - \
        add_cmds "bash ./ec-mlp.sh" - \
        make $EC_DIR/ec-mlp-train-{i}.slurm  --concurrency 4

    touch $EC_DIR/ec-mlp-setup.done
}

cd $EC_DIR
omb job slurm submit "$EC_DIR/ec-mlp-train*.slurm" --max_tries 2 --wait --recovery $EC_DIR/slurm-recovery.json

# step 5: explore 

LMP_DIR=$ITER_DIR/lammps
mkdir -p $LMP_DIR

[ -f $LMP_DIR/setup.done ] && echo "$SYS_NAME: skip lammps setup" || {
    for SYS_NAME in $SYS_NAMES; do
        CHARGE=${CHARGE_MAP[$SYS_NAME]}
        echo "Processing $SYS_NAME with CHARGE=$CHARGE"
        LMP_TASK_DIR=$LMP_DIR/task/$SYS_NAME
        mkdir -p $LMP_TASK_DIR
        EX_DATA_DIR=$LMP_DIR/lammps-data/$SYS_NAME
        mkdir -p $EX_DATA_DIR
        ai2-kit tool ase read $SYS_EXPLORE_DIR/$SYS_NAME/*.xyz - write_dplr_lammps_data $EX_DATA_DIR/{i:03d}.data --type_map [Cu,H,O] --sel_type [2] --sys_charge_map [0,1,6] --model_charge_map [-8]
    
        omb combo \
            add_files DATA_FILE "$EX_DATA_DIR/*" --abs - \
            add_file_set EC_MODELS "$EC_DIR/ec-mlp-model-*/ec-mlp-compress.pb" --abs - \
            add_var TEMP 330 430 530 - \
            add_var STEPS $EXPLORE_STEPS - \
            add_var CHARGE "$CHARGE" - \
            add_var FREQ $FREQ - \
            add_var EC_SINGLE_MODEL $EC_DIR/ec-mlp-model-0/ec-mlp-compress.pb - \
            add_randint SEED -n 10000 -a 0 -b 99999 - \
            set_broadcast SEED - \
            make_files $LMP_TASK_DIR/job-{TEMP}K-{i:03d}/input.lmp --template $CONFIG_DIR/lammps/input.lmp - \
            make_files $LMP_TASK_DIR/job-{TEMP}K-{i:03d}/run.sh    --template $CONFIG_DIR/lammps/run.sh --mode 755 - \
            done
    done
    omb batch \
        add_work_dirs "$LMP_DIR/task/*/job-*" - \
        add_header_files $CONFIG_DIR/lammps/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $LMP_DIR/task/lammps-{i}.slurm  --concurrency 20
    
    touch $LMP_DIR/setup.done
}
cd $LMP_DIR/task
omb job slurm submit "$LMP_DIR/task/lammps*.slurm" --max_tries 2 --wait --recovery $LMP_DIR/slurm-recovery.json


# Step 6: Screening
SCREENING_DIR="$ITER_DIR/screening"
mkdir -p "$SCREENING_DIR"

if [ -f "$SCREENING_DIR/screening.done" ]; then
    echo "skip screening"
else
    for SYS_NAME in $SYS_NAMES; do
        echo "Processing screening for system: $SYS_NAME"
        SCREENING_TASK_DIR="$SCREENING_DIR/$SYS_NAME"
        mkdir -p "$SCREENING_TASK_DIR"
        
        # 1. Perform overall model deviation analysis and classify structures for all jobs
        ai2-kit tool model_devi \
            read "$LMP_DIR/task/$SYS_NAME/job-*/" --traj_file dump.lammpstrj --md_file model_devi.out --specorder "[Cu,H,O]" --ignore_error - \
            grade --lo 0.20 --hi 0.40 --col max_devi_f - \
            dump_stats "$SCREENING_TASK_DIR/stats.tsv" - \
            write "$SCREENING_TASK_DIR/good.xyz"   --level good - \
            write "$SCREENING_TASK_DIR/decent.xyz" --level decent - \
            write "$SCREENING_TASK_DIR/poor.xyz"   --level poor
            
        # 2. Update explore_init_str (Extract new training sets specifically from 330K jobs)
        echo "Updating explore_init_str for 330K jobs..."
        > "$SCREENING_TASK_DIR/update.xyz"  # Initialize an empty master target file
        
        for i in "$LMP_DIR"/task/"$SYS_NAME"/job-330K*/; do
            # 2.1 Find structures with deviation between 0.1-0.3 in the current trajectory and save them inplace as decent_update.xyz
            ai2-kit tool model_devi \
                read "$i" --traj_file dump.lammpstrj --md_file model_devi.out --specorder "[Cu,H,O]" --ignore_error - \
                grade --lo 0.10 --hi 0.30 --col max_devi_f - \
                write "decent_update.xyz" --inplace -
            
            # 2.2 If decent_update.xyz is successfully generated and not empty, randomly sample 1 structure and append it to the master file
            if [ -s "$i/decent_update.xyz" ]; then
                ai2-kit tool ase \
                    read "$i/decent_update.xyz" - \
                    sample 1 -method random - \
                    write "$SCREENING_TASK_DIR/update.xyz" --append
                
                # Remove intermediate file to keep the directory clean
                rm -f "$i/decent_update.xyz"
            fi
        done
        
        # 3. Clean up empty files to prevent downstream errors
        [ ! -s "$SCREENING_TASK_DIR/decent.xyz" ] && rm -f "$SCREENING_TASK_DIR/decent.xyz"
        [ ! -s "$SCREENING_TASK_DIR/update.xyz" ] && rm -f "$SCREENING_TASK_DIR/update.xyz"
    done
    
    touch "$SCREENING_DIR/screening.done"
fi

# Print statistics for each system
echo -e "\n=== Screening Stats ==="
for SYS_NAME in $SYS_NAMES; do
    if [ -f "$SCREENING_DIR/$SYS_NAME/stats.tsv" ]; then
        echo "Stats for $SYS_NAME:"
        cat "$SCREENING_DIR/$SYS_NAME/stats.tsv"
        echo "-----------------------"
    fi
done

# Mark iteration as done
touch "$ITER_DIR/iter.done"
