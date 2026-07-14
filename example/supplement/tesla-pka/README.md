# TESLA Workflow for pKa
A bash-script based TESLA workflow for pKa calculation with a FEP-TI based active-learning scheme.

## Introduction
This example follows the same workflow construction style as `tesla-h2o`, using bash scripts to organize the TESLA active-learning loop.

The target here is pKa calculation, and the method is based on a FEP-TI workflow. The main differences in `tesla-pka` are:

* dual-state `DeepMD` screening for initial and final states
* `lambda`-dependent `LAMMPS` sampling for FEP-TI style exploration
* paired `CP2K` labeling for `ini` and `fin` structures
* optional update of the protonated initial configurations used in later MD exploration

## Getting Started
To run the workflow:

```bash
./run.sh
```

By default, `run.sh` launches five active-learning iterations with progressively larger training and MD budgets.

## Notes on the pKa Method
This workflow is not a standard single-state active-learning loop. It is designed for proton dissociation, so the exploration and labeling stages must handle both the protonated and deprotonated states consistently.

In `LAMMPS`, the sampling is built on two `DeepMD` potentials and a `lambda`-dependent interpolation between them:

```lammps
pair_style hybrid/overlay &
           deepmd @DP_MODELS out_freq ${SAMPLE_FREQ} out_file model_devi_ini.out &
           deepmd @DP_MODELS out_freq ${SAMPLE_FREQ} out_file model_devi_fin.out
pair_coeff * * deepmd 1 O H H
pair_coeff * * deepmd 2 O H NULL

fix  sampling_PES all adapt 0 &
     pair deepmd:1 scale * * v_LAMBDA_i &
     pair deepmd:2 scale * * v_LAMBDA_f
```

This means each MD job is not sampling one fixed PES. It is sampling along an alchemical path that connects the protonated and deprotonated states, while monitoring model deviation for both states separately.

Accordingly, each iteration expands MD jobs over `lambda` as well as temperature:

```bash
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
    done
```

The labeling stage is also state-aware. Candidate structures are labeled in both protonated and deprotonated forms; for the deprotonated state, the target proton is removed before `CP2K` calculation, and the electronic charge must be assigned consistently for the corresponding state. As a result, the workflow accumulates two kinds of reference data, one for the initial state and one for the final state, instead of treating all sampled structures as a single pool.

The screening stage follows the same idea. Model deviation is not evaluated once on a single trajectory output; instead, the workflow screens the protonated and deprotonated states separately, so uncertain configurations can be collected for both ends of the alchemical path before labeling.

The labeling stage is also split into protonated and deprotonated branches. For the deprotonated branch, the target proton is removed before `CP2K` calculation, and the electronic charge is assigned accordingly. The generated reference data therefore remain consistent with the two end states used in FEP-TI sampling.

The main point is that this method learns not only stable configurations, but also the consistency between two charge/protonation states and the alchemical region connecting them. That is the key feature that makes it suitable for FEP-TI based pKa active learning.

## Project Layout
* `00-config`: template files for `DeepMD`, `LAMMPS`, `CP2K`, and Slurm job headers
* `10-data`: initial DeepMD datasets and initial `LAMMPS` structures
* `20-workflow`: the scripted FEP/pKa active-learning workflow
* `30-workdir`: generated during runtime, contains per-iteration working directories
* `run.sh`: main entry point for the full workflow

You can also customize the workflow by:

* modifying configuration in the `00-config` folder
* modifying training strategy by editing `20-workflow/iter-fep-pka-dp-lammps-cp2k.sh`
* adding more iterations by editing `run.sh`

Send PR if you have any good ideas to improve the workflow.
