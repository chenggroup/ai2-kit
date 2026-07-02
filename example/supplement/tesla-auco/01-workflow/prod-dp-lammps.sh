#!/bin/bash
set -eu

# ensure the following environment variables are set
omb shell require-env WORK_DIR CONFIG_DIR PROD_DIR \
                      DP_MODELS MD_STEPS MD_TEMP

mkdir -p $PROD_DIR

[ -f $PROD_DIR/setup.done ] && echo "skip lammps setup" || {
    omb combo \
        add_files DATA_FILE $CONFIG_DIR/plumed/lmp_4.data --abs - \
        add_file_set DP_MODELS $DP_MODELS --abs - \
        add_var TEMP $MD_TEMP  - \
        add_var STEPS $MD_STEPS - \
        add_var SAMPLE_FREQ $SAMPLE_FREQ - \
        add_randint SEED -n 10000 -a 0 -b 99999 --uniq - \
        set_broadcast SEED - \
        compute BIASFACTOR "25000 // TEMP" - \
        make_files $PROD_DIR/job-{TEMP}K/lammps.in --template $CONFIG_DIR/lammps/lammps.in - \
        make_files $PROD_DIR/job-{TEMP}K/plumed.in --template $CONFIG_DIR/plumed/prod.plumed - \
        make_files $PROD_DIR/job-{TEMP}K/run.sh    --template $CONFIG_DIR/lammps/run.sh --mode 755 - \
        run_cmd "echo {TEMP} > $PROD_DIR/job-{TEMP}K/TEMP" - \
        done

    omb batch \
        add_work_dirs "$PROD_DIR/job-*" - \
        add_header_files $CONFIG_DIR/lammps/slurm-header.sh - \
        add_cmds "bash ./run.sh" - \
        make $PROD_DIR/lammps-{i}.slurm

    touch $PROD_DIR/setup.done
}


[ -f $PROD_DIR/run.done ] && echo "skip lammps run" || {
    omb job slurm submit "$PROD_DIR/lammps*.slurm" --max_tries 2 --wait --recovery $PROD_DIR/slurm-recovery.json
    touch $PROD_DIR/run.done
}
