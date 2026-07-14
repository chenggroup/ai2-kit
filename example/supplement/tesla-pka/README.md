# TESLA Workflow for pKa
A bash script based workflow for training a machine learning potential automatically for proton transfer and pKa free energy perturbation workflows.

## Introduction
**T**rain-**E**xplore-**S**creen-**L**abel **A**ctive-learning (TESLA) workflow is a bash script based workflow for training a machine learning potential automatically.

This workflow is inspired by [dpgen](https://github.com/deepmodeling/dpgen) and [ai2-kit](https://github.com/chenggroup/ai2-kit).

It is a bash script built with `oh-my-batch` and `ai2-kit`, which makes it easy to customize.
Developers can easily add their own steps to the workflow by modifying the bash script directly.

This `tesla-pka` example targets a proton dissociation setting. Compared with the basic water example, it introduces:

* dual-state `DeepMD` screening for initial and final states
* `lambda`-dependent `LAMMPS` sampling for free energy perturbation style exploration
* paired `CP2K` labeling for `ini` and `fin` structures
* optional update of the protonated initial configurations used in later MD exploration

## Getting Started
To run the workflow, you need to ensure your environment has Python 3. And then all you need to do is to run the following command:

```bash
./run.sh
```

The script installs the required Python packages:

* `ai2-kit>=1.0.8`
* `oh-my-batch>=0.7.2`

By default, `run.sh` launches five active-learning iterations with progressively larger training and MD budgets.

## Workflow Overview
Each iteration in `20-workflow/iter-fep-pka-dp-lammps-cp2k.sh` contains four stages:

* `training`: train an ensemble of `DeepMD` models from `10-data/deepmd-init` and previously labeled datasets in `30-workdir/iter-*/new-dataset-*`
* `explore`: run `LAMMPS` jobs over multiple temperatures and `LAMBDA_f` values using protonated `h3o` initial structures from `10-data/lammps-init/h3o`
* `screening`: use `ai2-kit tool model_devi` to screen both `model_devi_ini.out` and `model_devi_fin.out`
* `labeling`: sample candidate structures, remove the target proton for final-state jobs, and label them with `CP2K`

At the end of each iteration, the `CP2K` outputs are converted into new DeepMD datasets:

* `new-dataset-ini`
* `new-dataset-fin`

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

The main point is that this method learns not only stable configurations, but also the consistency between two charge/protonation states and the alchemical region connecting them. That is the key feature that makes it suitable for pKa-related active learning.

## Project Layout
* `00-config`: template files for `DeepMD`, `LAMMPS`, `CP2K`, and Slurm job headers
* `10-data`: initial DeepMD datasets and initial `LAMMPS` structures
* `20-workflow`: the scripted FEP/pKa active-learning workflow
* `30-workdir`: generated during runtime, contains per-iteration working directories
* `run.sh`: main entry point for the full workflow

## Important Parameters
You can customize the workflow by editing `run.sh`. Important variables include:

* `ATOMS_TO_REMOVE`: 1-based atom id of the proton removed in final-state labeling
* `MODEL_NUM`: number of independently trained `DeepMD` models
* `MD_TEMP`: temperatures used in `LAMMPS` exploration
* `LAMBDA_f`: final-state coupling parameters used during FEP exploration
* `MODEL_DEVI_COND`: lower and upper thresholds for model deviation screening
* `MAX_LABEL`: maximum number of candidate structures labeled for each state per iteration
* `USE_BAD_CONFS`: number of poor configurations optionally added to labeling
* `UPDATE_MD_CONFS`: number of candidate initial structures used to refresh MD starting points

You can also customize the workflow by:

* modifying configuration in the `00-config` folder, which includes template files of `DeepMD`, `LAMMPS`, `CP2K`, `Slurm`, etc.
* modifying training strategy by editing `20-workflow/iter-fep-pka-dp-lammps-cp2k.sh`
* adding more iterations by editing `run.sh`

Send PR if you have any good ideas to improve the workflow.
