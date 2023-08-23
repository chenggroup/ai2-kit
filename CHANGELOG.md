# Change Logs

## v0.6.5
* fix: cp2k input file generation bug

## v0.6.4
* feat: [cll-workflow] support wave function warmup for cp2k, more detail [here](./example/config/cll-mlp-training/label-cp2k.yml)
* improvement: [cll-workflow] raise invalid artifact error earlier

## v0.6.3
* fix: ase tool arguments

## v0.6.2
* improvement: [cll-workflow] save good/decent/poor structures in xyz file.
* feat: implement dpdata tool, more details [here](./doc/manual/dpdata.md)

## v0.6.1
* fix: data handling bug

## v0.6.0
* feat: Support training DP model for FEP simulation
* feat: ase tool support delete_atoms
* **BREAKING CHANGE**: [cll-workflow] LAMMPS configuration has been change, [see example](./example/config/cll-mlp-training/explore-lammps.yml)

## v0.5.5
* improvement: change the squeue polling command
* feat: support `load_text` and `load_yaml` tag in yaml parser
* doc: config samples for cll-mlp-training workflow

## v0.5.4
* fix: model_devi_decent.xyz not created when no dumps are selected

## v0.5.3
* fix: atoms from lammps-dump-text not ordered by id

## v0.5.2
* fix: sort atoms before exporting as POSCAR

## v0.5.1
* improvement: suppress numba warning
* improvement: checkpoint rm support exclude pattern

## v0.5.0
* **BREAKING CHANGE**: [cll-workflow] checkpoint file may not compatible with < v0.5.0
* feat: select distinct structures by global descriptor grouping

## v0.4.0
* **BREAKING CHANGE**: [cll-workflow] checkpoint file may not compatible with < v0.4.0
* **BREAKING CHANGE**: [cll-workflow] config file change: `select.by_threshold` => `select.model_devi`
* feat: [cll-workflow] support explore with LASP
* refactor: [cll-workflow] data conversion system
* chore: asap tool integration

## v0.3.33
* refactor: list_sample method
* fix: use len(dp_system) > 0 to filter broken data

## v0.3.32
* fix: ignore data merge when SCF is not converged

## v0.3.31
* refactor: simplify data handling

## v0.3.30
* fix: slurm job checkpoint key

## v0.3.29
* fix: cp2k command line

## v0.3.28
* fix: use override instead of append for cp2k output

## v0.3.26
* improvement: checkpoint tool command line
* fix: slurm job checkpoint key

## v0.3.25
* improvement: pretty table output

## v0.3.24
* fix: create missing dir

## v0.3.23
* fix: cpk2 command

## v0.3.22
* improvement: cp2k support post_cp2k_cmd option
* improvement: selector will save result in csv format

## v0.3.21
* fix: dp empty validation data error

## v0.3.20
* doc: generate online doc
* chore: github workflow
* improvement: support random sampling if selected structures hit limitation
* fix: add type_map when converting data to deepmd/npy

## v0.3.19
* feat: distance analysis for proton transfer

## v0.3.18
* fix: convert vasp data for deepmd

## v0.3.17
* improvement: sample list evenly instead of truncate when selecting new structures

## v0.3.16
* fix: use max_devi_f instead of avg_devi_f to select structures

## v0.3.15
* fix: label all init data

## v0.3.14
* feat: label all init data

## v0.3.13
* feat: support specify deepmd validation data

## v0.3.12
* fix: model deviation selection rules
* improvement: cll: stop workflow if no more data is generated
* refactor module names

## v0.3.11
* fix: pass down ancestor attributes to child artifacts

## v0.3.10
* chore: remove cp2k-input-tools
* feat: implement cpk2 parse module

## v0.3.9
* fix: raise error on job failure

## v0.3.8
* feat: ase tool support set_cell and set_pbc commands

## v0.3.7
* feat: support VASP in cll training workflow
* feat: support `compress_model` option in deepmd step

## v0.3.6
* feat: implement read tag for yaml parser

## v0.3.5
* feat: support plumed config in cll workflow
* feat: support artifacts attributes inheritance in workflow
* feat: implement join tag for yaml parser

## v0.3.4
* feat: apply checkpoint to all workflows and job submission steps
* feat: add checkpoint command line interface

## v0.3.3
* refactor: using async.io to implement tasks and workflows

## v0.3.2
* improvement: support relative work_dir

## v0.3.1
* feat: async job polling

## v0.3.0
* feat: ase toolkit
* fix: fep training workflow
* improvement: speed up command line interface with lazy import

## v0.2.0
* feat: proton transfer analysis toolkit

## v0.1.0
* feat: cll-mlp-training workflow
* feat: fep-mlp-training workflow

## v0.0.2
* feat: support providing initial structures for label task at first iteration.
* feat: support dynamic configuration update for each iterations.
* improvement: reduce size of remote python script by compressing it with bz2.
* fix: ssh connection closed when running large remote python script.
* refactor: design common interface for CLL workflow.
* refactor: use ase.Atoms as the universal in-memory data structure.
* refactor: use ResourceManager to manage executors and artifacts.

## v0.0.1
* feat: utilities to execute and manage jobs in local or remote HPC job scheduler. 
* feat: utilities to run python functions in remote machines directly.
* feat: utilities to simplified automated workflows development with reusable components.
* feat: fundamental CLL & FEP workflows.