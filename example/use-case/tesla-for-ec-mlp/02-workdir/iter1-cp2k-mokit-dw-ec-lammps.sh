#!/bin/bash

set -e

# ensure ITER_NAME is set
[ -z "$ITER_NAME" ] && echo "environment variable ITER_NAME is not set" && exit 1 || echo "ITER_NAME=$ITER_NAME"
[ -z "$Labeling_num" ] && echo "environment variable Labeling_num is not set" && exit 1 || echo "Labeling_num=$Labeling_num"
[ -z "$TRAINING_STEPS" ] && echo "environment variable TRAINING_STEPS is not set" && exit 1 || echo "TRAINING_STEPS=$TRAINING_STEPS"
[ -z "$EXPLORE_STEPS" ] && echo "environment variable EXPLORE_STEPS is not set" && exit 1 || echo "EXPLORE_STEPS=$EXPLORE_STEPS"
[ -z "$FREQ" ] && echo "environment variable FREQ is not set" && exit 1 || echo "FREQ=$FREQ"
[ -z "$UPDATE_INITIAL_STR" ] && echo "environment variable UPDATE_INITIAL_STR is not set" && exit 1 || echo "UPDATE_INITIAL_STR=$UPDATE_INITIAL_STR"

CONFIG_DIR=/public/home/jpqiu/10_interface/mlp/1_Cu100/0_cl_cover/00-config
WORK_DIR=/public/home/jpqiu/10_interface/mlp/1_Cu100/0_cl_cover/02-workdir
INIT_LABELING_DIR=/public/home/jpqiu/10_interface/mlp/1_Cu100/0_cl_cover/01-init/00-data/0815/label
SYS_EXPLORE_DIR=/public/home/jpqiu/10_interface/mlp/1_Cu100/0_cl_cover/01-init/01-explore/0902
SYS_NAMES="1cl 1k 2k"
#SYS_NAMES="Cu96H184O92 Cu96Cl16H168O84 Cu96Cl16H166O83K2 Cu96Cl16H164O82K4 Cu96Cl16H166O83F2 Cu96Cl16H162O81F4"
# 定义金属电荷
declare -A CHARGE_MAP=(
    [1cl]=18
    [1k]=14
    [2k]=12
)
    #[Cu96Cl16H162O81F4]=20



# create iter dir
ITER_DIR=$WORK_DIR/iter-$ITER_NAME
mkdir -p $ITER_DIR

[ -f $ITER_DIR/iter.done ] && echo "iteration $ITER_NAME already done" && exit 0 || echo "starting iteration at $ITER_DIR"

PREV_ITER_NAME=$(printf "%03d" $((10#$ITER_NAME - 1)))
PREV_ITER_DIR=$WORK_DIR/iter-$PREV_ITER_NAME


# step 1: labeling  ---energy and force
LABELING_DIR=$ITER_DIR/cp2k
mkdir -p $LABELING_DIR

[ -f $LABELING_DIR/cp2k-setup.done ] && echo "$SYS_NAME: skip cp2k setup" || {
    # 每个体系取多少个labeling
    ai2-kit tool ase read $PREV_ITER_DIR/screening/*/decent.xyz - sample $Labeling_num - \
        write_frames $LABELING_DIR/data/{i:03d}.inc --format cp2k-inc

    for f in $LABELING_DIR/data/*.inc; do
        sed -i '/^ *X /d' "$f"
    done

    omb combo \
        add_files DATA_FILE "$LABELING_DIR/data/*" --abs - \
        make_files $LABELING_DIR/job-{i:03d}/cp2k.inp --template $CONFIG_DIR/cp2k/cp2k.inp - \
        make_files $LABELING_DIR/job-{i:03d}/cp2k.sh   --template $CONFIG_DIR/cp2k/cp2k.sh --mode 755 - \
        make_files $LABELING_DIR/job-{i:03d}/wc.py   --template $CONFIG_DIR/mokit/wc.py --mode 755 - \
        make_files $LABELING_DIR/job-{i:03d}/wc.sh   --template $CONFIG_DIR/mokit/wc.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$LABELING_DIR/job-*" - \
        add_header_files $CONFIG_DIR/cp2k/slurm-header.sh - \
        add_cmds "bash ./cp2k.sh" - \
        make $LABELING_DIR/cp2k-{i}.slurm  --concurrency 60

    touch $LABELING_DIR/cp2k-setup.done
}
cd $LABELING_DIR
omb job slurm submit "$LABELING_DIR/cp2k-*.slurm" --max_tries 2 --wait --recovery $LABELING_DIR/cp2k-slurm-recovery.json

# step 2: labeling  ---wannier center
[ -f $LABELING_DIR/wc-setup.done ] && echo "$SYS_NAME: skip wc setup" || {
    omb batch \
        add_work_dirs "$LABELING_DIR/job-*" - \
        add_header_files $CONFIG_DIR/mokit/slurm-header.sh - \
        add_cmds "bash ./wc.sh" - \
        make $LABELING_DIR/wc-{i}.slurm  --concurrency 60
    
    touch $LABELING_DIR/wc-setup.done
    }

cd $LABELING_DIR
omb job slurm submit "$LABELING_DIR/wc*.slurm" --max_tries 2 --wait --recovery $LABELING_DIR/wc-slurm-recovery.json



# step 3: training dw model

DW_DIR=$ITER_DIR/dw
mkdir -p $DW_DIR
DECAY=$((TRAINING_STEPS / 200))

[ -f $DW_DIR/dw-setup.done ] && echo "skip dw setup" || {
    # convert cp2k output and wannier center to dpdata
    ai2-kit tool dpdata read $ITER_DIR/cp2k/job-*/ --fmt cp2k/dplr --cp2k_output="output" --wannier_file="wannier.xyz" --type_map="[Cu,Cl,H,O,K]" --sel_type="[1,3,4]" --model_charge_map="[-8,-8,-8]" - write $ITER_DIR/new-dataset
    # generate 1 random seed to train 1 dw models
    omb combo \
        add_randint SEED -n 1 -a 0 -b 999999 --uniq - \
        add_var STEPS $TRAINING_STEPS - \
        add_var DECAY $DECAY - \
        add_file_set DP_DATASET  "$WORK_DIR/iter-*/new-dataset/*" --format json-item --abs - \
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
    # generate 4 uniq random seed to train 4 deepmd models
    omb combo \
        add_randint SEED -n 4 -a 0 -b 999999 --uniq - \
        add_var STEPS $TRAINING_STEPS - \
        add_var DECAY $DECAY - \
        add_file_set DW_MODEL  "$DW_DIR/dw-model-*/dw.pb" --format json-item --abs - \
        add_file_set DP_DATASET   "$WORK_DIR/iter-*/new-dataset/*" --format json-item --abs - \
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
        if [ "$UPDATE_INITIAL_STR" == "TRUE" ]; then
            ai2-kit tool ase read $PREV_ITER_DIR/screening/$SYS_NAME/update.xyz - write_dplr_lammps_data $EX_DATA_DIR/{i:03d}.data --type_map [Cu,Cl,H,O,K] --sel_type [1,3,4] --sys_charge_map [0,7,1,6,9] --model_charge_map [-8,-8,-8]
        else
            ai2-kit tool ase read $SYS_EXPLORE_DIR/$SYS_NAME/*.xyz - write_dplr_lammps_data $EX_DATA_DIR/{i:03d}.data --type_map [Cu,Cl,H,O,K] --sel_type [1,3,4] --sys_charge_map [0,7,1,6,9] --model_charge_map [-8,-8,-8]
        fi
    
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
        make $LMP_DIR/task/lammps-{i}.slurm  --concurrency 6
    
    touch $LMP_DIR/setup.done
}
cd $LMP_DIR/task
omb job slurm submit "$LMP_DIR/task/lammps*.slurm" --max_tries 2 --wait --recovery $LMP_DIR/slurm-recovery.json


# step 6: screening
SCREENING_DIR=$ITER_DIR/screening
mkdir -p $SCREENING_DIR
[ -f $SCREENING_DIR/screening.done ] && echo "skip screening" || {
    for SYS_NAME in $SYS_NAMES; do
        SCREENING_TASK_DIR=$SCREENING_DIR/$SYS_NAME
        mkdir -p $SCREENING_TASK_DIR
        num_dirs=$(find "$LMP_DIR/task/$SYS_NAME/job-330K*/" -type d | wc -l)
        ai2-kit tool model_devi \
            read "$LMP_DIR/task/$SYS_NAME/job-*/" --traj_file dump.lammpstrj --md_file model_devi.out --specorder "[Cu,Cl,H,O,K]" --ignore_error - \
            grade --lo 0.25 --hi 0.60 --col max_devi_f - \
            dump_stats $SCREENING_TASK_DIR/stats.tsv - \
            write $SCREENING_TASK_DIR/good.xyz   --level good - \
            write $SCREENING_TASK_DIR/decent.xyz --level decent - \
            write $SCREENING_TASK_DIR/poor.xyz   --level poor - \
            done
        #Update explore_init_str
      #counter=0 
      #for i in $LMP_DIR/task/$SYS_NAME/job-330K*/; do
      #    echo "$i"
      #    ai2-kit tool model_devi \
      #        read "$i" --traj_file dump.lammpstrj --md_file model_devi.out --specorder "[Cu,Cl,H,O,K]" --ignore_error - \
      #        grade --lo 0.11 --hi 0.30 --col max_devi_f - \
      #        write "$SCREENING_TASK_DIR/update_${counter}.xyz" --level decent - \
      #    done
      #    ai2-kit tool ase read "$SCREENING_TASK_DIR/update_${counter}.xyz" - sample 1 -method random - write "$SCREENING_TASK_DIR/update_${counter}.xyz"
      #    counter=$((counter + 1))
      #done
      ## 合并所有 update_${counter}.xyz，并处理删除X、更新原子数
      #> "$SCREENING_TASK_DIR/update.xyz"
      #for f in "$SCREENING_TASK_DIR"/update_*.xyz; do
      #    awk 'NR==1{n=0; next} NR==2{h=$0; next} $1!="X"{a[++n]=$0} END{print n; print h; for(i=1;i<=n;i++) print a[i]}' "$f" >> "$SCREENING_TASK_DIR/update.xyz"
      #done
      #     # 如果 decent.xyz 是空的就删掉
      # [ ! -s "$SCREENING_TASK_DIR/decent.xyz" ] && rm -f "$SCREENING_TASK_DIR/decent.xyz"
    done
        
    touch $SCREENING_DIR/screening.done
}
#cat $SCREENING_DIR/stats.tsv


# mark iteration as done
touch $ITER_DIR/iter.done
