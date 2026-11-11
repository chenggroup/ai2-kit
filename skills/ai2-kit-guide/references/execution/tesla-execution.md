# TESLA stage contracts and execution checks

These instructions apply to active learning. For fitting existing labels only, use the supervised route in [the execution module](README.md). The commands below illustrate DeepMD/LAMMPS/CP2K; use [engine adaptation](engine-adaptation.md) for other native commands, templates and formats. Read the selected revision's complete iteration script before changing it. There is no universal CLI shared by all trainers.

## Initial labels and independent splits

Record label provenance and DFT settings, file hashes, frame identities, units and split rules. AIMD frames are usually correlated: prefer independent trajectories/states or separated time blocks. Randomly selecting adjacent frames does not make them independent.

`AseTool.slice` selects frames; `DpdataTool.slice` selects systems. Do not use the latter as if it split frames after conversion. Convert each already-separated labelled extxyz split independently, substituting real paths and element names:

```bash
ai2-kit tool ase read /data/train.extxyz --format extxyz - \
  to_dpdata --labeled --type_map '[ElementA,ElementB]' - write /project/data/train
```

Check `type.raw`, `type_map.raw`, `set.*/{coord,energy,force,box}.npy` and frame counts. Ordinary MLFF training requires energies and forces. When using virial/stress losses, additionally verify units, sign and volume conventions. Do not silently drop extra inputs needed by a specialised model.

## Initialisation and templates

`run.sh` defines CONFIG_DIR/WORK_DIR/TYPE_MAP and iteration parameters; `01-workflow/setup.sh` prepares the initial training data and exploration structures. Replace example AIMD/species inputs; an example's 50-frame sample is not a universal recipe. Reserve validation/test data before initialising training.

Adapt the following together:

- Trainer: architecture, type map, cutoff, loss/virial terms, steps, save intervals, seeds and export format.
- MD: types/masses, model files, units/PBC, ensemble, temperature/pressure, timestep, enhanced sampling and output intervals.
- DFT: species/basis/pseudopotentials, XC/dispersion, cutoffs, k points, charge/spin, SCF and single-point E/F outputs.
- Launch scripts for each stage: scheduler partition/account, resources, modules/environments, MPI/threads and executable paths.

## Each iteration

| Stage | Inputs and operations | Evidence before continuing |
| --- | --- | --- |
| Train | Expand seeds, steps and training-only dataset combinations; package and submit tasks | All requested committee members completed, independent seeds, loadable models and recorded formats/hashes |
| Explore | Render jobs from models, initial structures and MD settings | Nonempty plausible trajectories, aligned frame/step deviations and consistent units |
| Screen | Use the installed `model_devi` read/slice/grade/statistics/export interface | Actual read/discard counts, good/candidate/poor statistics and exported candidates |
| Label | Render reference inputs, submit/wait, parse converged outputs | Correct E/F, cell and atom mapping; traceable successes and failures |
| Accumulate | Convert with the appropriate ASE/dpdata parser and append new labels | Nonempty valid labels, unchanged reference method and no test contamination |

A representative Slurm submission is:

```bash
omb job slurm submit '/project/20-workdir/iter-001/deepmd/dp-train*.slurm' \
  --max_tries 2 --wait --recovery /project/20-workdir/iter-001/deepmd/slurm-recovery.json
```

This submits real jobs; use it only after project, host, resources and scope are configured and authorised. On a non-Slurm host replace submission AND waiting/state logic, not merely the scheduler command.

## Source-derived pitfalls

- Some example setup scripts read `.xyz` without `--format extxyz`. Specify labelled reading in the project copy and verify the converted labels.
- DeepMD templates may contain `@SEED`, `@TRAIN_STEPS`, `@DECAY_STEPS` and `@DP_DATASET`; they are not valid rendered JSON yet.
- The original DeepMD scripts train/freeze/compress `.pb` models and exploration reads `compress.pb`. Backend changes require compatible schemas, CLI, export/compression and MD plugins, not just a filename extension.
- Under `set -eu`, an optional `[ "$OPTION" -gt 0 ] && { ...; }` at the end of a function/script can propagate false status. Use explicit `if ...; then ...; fi` and check actual exit behaviour.
- Some no-candidate branches run `touch iter.done; exit 1`, which stops a `set -e` parent. Make the stop reason explicit; exclude empty trajectories, all-poor exploration and dropped frames before interpreting it as convergence.
- `--ignore_error` skips failed files. If partial labelling failure is allowed, preserve task/success/failure lists and a minimum valid-label requirement; do not silently declare the round successful.
- The MACE example's POTCAR is a placeholder. Use lawfully available real pseudopotentials and verify species order. Replace broad merge globs and `|| true` with verified file lists and label/count checks.
- Non-native committee deviations require consistent eV/angstrom units, atom order, frames and model membership. Mark unavailable virials as unavailable rather than treating dummy columns as physical screening data.
- Changing inputs in a marker-based project invalidates dependent artifacts. Removing only the final marker is insufficient; prefer a fresh run directory and preserve failed logs.

## Budget, stopping and final models

Record committee size, iteration limit, MD steps/tasks, DFT label cap and queue limits before execution. Respect a user-defined passing-rate rule; reaching the iteration cap is a budget stop, not proof of scientific convergence.

Labels from the final labelling stage are not included in that iteration's earlier training. Retrain/export/test if the delivered models must include all accumulated training labels. Keep the final test data out of tuning, report composition-specific errors and aggregation weights, and record prediction tests, any requested physical validation and the actual stop reason.
