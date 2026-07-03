# TESLA Workflow
A bash script based workflow for training a machine learning potential automatically.

## Introduction
**T**rain-**E**xplore-**S**creen-**L**abel **A**ctive-learning (TESLA) workflow is a bash script based workflow for training a machine learning potential automatically. 

This workflow is inspired by [dpgen](https://github.com/deepmodeling/dpgen) and [ai2-kit](https://github.com/chenggroup/ai2-kit). 

It is a bash script built with `oh-my-batch` and `ai2-kit`, which makes it easy to customize. 
Developers can easily add their own steps to the workflow by modifying the bash script directly. 

Two workflow variants are provided:
- **MACE + OpenMM + VASP** (`iter-classic-mace-openmm-vasp.sh`) — the primary workflow configured in `run.sh`.
- **DeePMD + LAMMPS + CP2K** (`iter-classic-dp-lammps-cp2k.sh`) — the original reference workflow (retained for comparison).

## Getting Started
To run the workflow, you need to ensure your environment has Python 3. And then all you need to do is to run the following command:

```bash
./run.sh
```

### Manual environment steps before running
1. **MACE**: install `mace-torch` and `openmmml` (added to the `pip install` line in `run.sh`).
2. **VASP POTCAR**: generate `00-config/vasp/POTCAR` by concatenating the per-element POTCAR files in the correct element order (matching `TYPE_MAP`). See the placeholder file for instructions.
3. **Cluster modules**: edit the three `slurm-header.sh` files under `00-config/mace/`, `00-config/openmm/`, and `00-config/vasp/` to load the correct environment modules for your HPC system.

To customize the workflow, you can:
* Modify configuration in `00-config` folder, which includes template files for `MACE`, `OpenMM`, `VASP`, and `Slurm`.
* Modify training strategy by copying `01-workflow/iter-classic-mace-openmm-vasp.sh` and adapting it.
* Add more iterations by editing `run.sh`.

Send PR if you have any good ideas to improve the workflow.

## Directory Structure

```
00-config/                  Configuration templates
  aimd.xyz                  AIMD trajectory used to seed initial training data
  mace/
    run.sh                  MACE training job template (@SEED, @TRAIN_EPOCHS, @TRAIN_FILE)
    slurm-header.sh         SLURM GPU header for MACE training jobs
  openmm/
    openmm-run.py           OpenMM NVT MD script using MACE potential via openmmml
    model-devi.py           Multi-model force-deviation calculator (outputs model_devi.out)
    run.sh                  OpenMM exploration job template
    slurm-header.sh         SLURM GPU header for OpenMM jobs
  vasp/
    INCAR                   VASP input parameters (single-point energy/forces)
    KPOINTS                 VASP k-point mesh (Gamma-only by default)
    POTCAR                  Placeholder — must be replaced by the user
    run.sh                  VASP labeling job template
    slurm-header.sh         SLURM CPU header for VASP jobs
  deepmd/                   DeePMD config templates (original workflow, retained)
  lammps/                   LAMMPS config templates (original workflow, retained)
  cp2k/                     CP2K config templates (original workflow, retained)

01-workflow/
  setup.sh                  Seed initial data for both workflow variants
  iter-classic-mace-openmm-vasp.sh   MACE/OpenMM/VASP TESLA iteration (active)
  iter-classic-dp-lammps-cp2k.sh     DeePMD/LAMMPS/CP2K TESLA iteration (reference)
  prod-dp-lammps.sh         Production MD script for DeePMD/LAMMPS

20-workdir/                 Runtime directory (created on first run; not version-controlled)
  setup.done
  mace-init-data/           Initial MACE training frames (extxyz, from aimd.xyz)
  openmm-data/              Initial OpenMM starting structures (extxyz, from aimd.xyz)
  dp-init-data/             Initial DeePMD training data (deepmd/npy, from aimd.xyz)
  lammps-data/              Initial LAMMPS structure files (from aimd.xyz)
  iter-001/ … iter-NNN/     Per-iteration work trees
    mace/                   MACE training models (mace_model.model per seed)
    openmm/                 OpenMM exploration job directories (traj.xyz, model_devi.out)
    screening/              Graded structures (good/decent/poor.xyz, stats.tsv)
    vasp/                   VASP labeling job directories (OUTCAR, success/error flags)
    new-dataset/            New training data in extxyz format for next iteration

run.sh                      Main workflow sequencer (MACE/OpenMM/VASP variant)
model-devi-plot.py          Utility: plot model deviation statistics
dp-test.py                  Utility: test DeePMD model accuracy
plot.sh / prod.sh           Production analysis helpers
```