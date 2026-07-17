# TESLA Workflow for Redox Potential
A bash-script based TESLA workflow for redox-potential calculation with a FEP-TI based active-learning scheme.

## Introduction
This example follows the same workflow construction style as `tesla-pka`, but targets redox chemistry instead of proton transfer.

The core feature of `tesla-redox` is that it must learn two electronic states at the same time:

* a neutral-state `DeepMD` committee
* a reduced-state `DeepMD` committee
* `lambda`-dependent `LAMMPS` sampling that interpolates between the two states
* paired `CP2K` labeling for neutral and reduced configurations

This makes the workflow different from a standard single-state TESLA loop. Exploration runs on a mixed potential energy surface, while screening, labeling, and dataset accumulation remain state-specific.

## Getting Started
To run the workflow:

```bash
./run.sh
```

By default, `run.sh` launches five active-learning iterations with progressively larger training and MD budgets.

## Notes on the Redox Method
The exploration stage uses two `DeepMD` committees in one `LAMMPS` job:

```lammps
pair_style hybrid/overlay &
           deepmd @DP_MODELS_NEU out_freq ${SAMPLE_FREQ} out_file model_devi_neu.out &
           deepmd @DP_MODELS_RED out_freq ${SAMPLE_FREQ} out_file model_devi_red.out

fix sampling_PES all adapt 0 &
    pair deepmd:1 scale * * v_LAMBDA_NEU &
    pair deepmd:2 scale * * v_LAMBDA_RED
```

So each MD trajectory samples an alchemical path between the neutral and reduced states, instead of running on a single fixed PES.

The workflow then screens model deviation separately for the two states and sends the resulting candidates to different `CP2K` templates:

* `cp2k-neu.inp` for the neutral branch
* `cp2k-red.inp` for the reduced branch

The generated reference data are written to two independent pools:

* `new-dataset-neu`
* `new-dataset-red`

These two pools are then reused in the next iteration to retrain the two `DeepMD` committees synchronously.

## Project Layout
* `00-config`: template files for `DeepMD`, `LAMMPS`, `CP2K`, and Slurm job headers
* `10-data`: initial DeepMD datasets and initial `LAMMPS` structures
* `20-workflow`: the scripted FEP/redox active-learning workflow
* `30-workdir`: generated during runtime, contains per-iteration working directories
* `run.sh`: main entry point for the full workflow

You can customize the workflow by:

* modifying configuration in the `00-config` folder
* modifying training and screening strategy in `20-workflow/iter-fep-redox-dp-lmp-cp2k.sh`
* adding more iterations by editing `run.sh`
