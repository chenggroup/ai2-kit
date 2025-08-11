# Change Logs

## v1.0.4
* implement lammps tool: analysis_density_convergence
* improve yaml tool

## v1.0.3
* implement lammps tool: gen_mass_map

## v1.0.2
* fix slice method in model_devi

## v1.0.1
* fix slice method in model_devi

## v1.0.0
* release version 1.0.0

## v0.25.4
* freeze numpy version to 1.24.3

## v0.25.3
* ase tool implement shuffle method

## v0.25.2
* fix code base status

## v0.25.1
* fix resolve conflict

## v0.25.0
* ase tool write support slice and chain option
* refine slice implementation in model_devi tool

## v0.24.4
* refine proton transfer Python API
* refine dplr data util

## v0.24.3
* upgrade fire-rpc to 0.1.3

## v0.24.2
* fix: the return data should be extxyz format

## v0.24.1
* feat: refine nmr predict command line

## v0.24.0
* feat: allow to run ai2-kit as an rpc server
* refine: command line options of nmrnet predict

## v0.23.6
* refactor slice implementation

## v0.23.5
* improvement: options to slice traj in model_devi

## v0.23.4
* fix: inplace should support ignore_error

## v0.23.3
* fix: model_devi read file

## v0.23.2
* improvement: options to ignore model_devi error

## v0.23.1
* fix: add model_devi tool to entry point

## v0.23.0
* breaking: modify model deviation tool

## v0.22.8
* fix: ignore error data in dpdata read

## v0.22.7
* fix: ignore error data in dpdata read

## v0.22.6
* fix: filter in model deviation tool
* fix: dplr in ase tool

## v0.22.5
* fix: make viber adapted for dp v3

## v0.22.4
* feat: support wannier charge != -8 (via `model_charge_map`)
* fix: broken dp_train_opts options

## v0.22.3
* fix: wrong filter in model devi tool

## v0.22.2
* feat: model deviation toolkit

## v0.22.1
* allow to override potential in lasp

## v0.22.0
* implement dplr and refactor dpff

## v0.21.4
* feat: handle cp2k/dplr in dpdata tool

## v0.21.3
* fix: dpdata tool wrong params

## v0.21.2
* fix: vasp wrong params

## v0.21.1
* fix: vasp wrong params

## v0.21.0
* improvement: ignore cutoff in viber
* improvement: vasp support attribute variable
* improvement: new ensable for lammps

## v0.20.2
* fix: pydantic 2 broken issue

## v0.20.1
* feat: extra options for dp train command

## v0.20.0
* add netCDF
* upgrade pydantic to 2.8.2

## v0.19.4
* fix: hpc tool typo
* feat: support use multiple gpus in the same batch job

## v0.19.3
* improve: dpdata: allow user to ignore broken data

## v0.19.2
* fix: dpdata eval wrong type mapping
* improve：add label to fes graph and dump result to json file

## v0.19.1
* fix cp2k-inc for ase write_frames
* fix out of date documents

## v0.19.0
* tesla: generate single dump file in lammps
* tool: refactor frame tool

## v0.18.5
* frame file process command line tool
* command line interface for reweighting algorithm

## v0.18.4
* deepmd: support `input_modifier_fn`

## v0.18.3
* lammps: use relative path to reference plumed input

## v0.18.2
* fix lammps data util for dplr
* fix cp2k cell digit limit

## v0.18.1
* refactor: ase write_each_frame to write_frames

## v0.18.0
* feat: reweighting algorithm
* feat: dpdata tool support eval
* improve: support covert between ase and dpdata format
* chore: replace `pylint` with `ruff`

## v0.17.5
* fix: redefine variable in for-loop

## v0.17.4
* fix: anyware execution

## v0.17.3
* fix: anyware output artifact

## v0.17.2
* fix: post_process_fn in anyware

## v0.17.1
* feat: cmd to write dplr lammps data

## v0.17.0
* feat: anyware for explore
* breaking change: new checkpoint system

## v0.16.1
* cp2k: use fp32 for input data
* lammps: support custom force field via `custom_ff` option

## v0.16.0
* ai2-cat: refine the basis set and potential selection
* ai2-cat: support multiple basic set and potential files
* ai2-spectr: fix wrong atom order

## v0.15.11
* fix hpc tool

## v0.15.10
* fix deepmd restart
* improve gen lammps script tool

## v0.15.9
* fix loading cp2k viber data

## v0.15.8
* fix ai2cat cp2k data path

## v0.15.7
* fix ai2spectr entry

## v0.15.6
* fix dpdata tool

## v0.15.5
* fix dpdata read

## v0.15.4
* fix fep pka bugs
* fix dpdata wrong arguments

## v0.15.0
* improve fep pka mode

## v0.14.11
* fix edge case of efield

## v0.14.9
* for dflow-galaxy: reference lammps data file use relative path

## v0.14.8
* feat: tool to format cp2k input

## v0.14.7
* hotfix: remove cp2k wfn if warmup step is failed

## v0.14.4
* hotfix: executor upload/download issue

## v0.14.2
* feat: temporary support for init from previous model

## v0.14.1
* refactor: make `asaplib` optional

## v0.14.0
* refactor: python remote execution solution
* feat: upload local python package to remote
* improvement: move some dependencies to optional 

## v0.13.2
* improvement: refactor slurm done file

## v0.13.1
* improvement: no need to return in wait method for slurm tool

## v0.13.0
* feat: `ai2-kit tool hpc` for managing HPC jobs

## v0.12.23
* improvement: add `ignore_error` for deepmd.

## v0.12.22
* improvement: add `ignore_error` for cp2k.

## v0.12.19
* fix: wrong lammps template for fep training
* improvement: select structure for next round from good and decent structures

## v0.12.9
* fix: wrong lammps template for fep training

## v0.12.8
* improvement: optimize job running indicator

## v0.12.7
* improvement: allow to override fix_statement and ensemble in artifact

## v0.12.6
* fix: rsplit max should be 1
* improve: don't keep good and poor structures to save storage

## v0.12.5
* feat: support max_decent_per_traj

## v0.12.4
* fix: wrong softlink path

## v0.12.3
* improvement: create softlink for dw_model

## v0.12.2
* fix: wrong lammps template_vars

## v0.12.1
* fix: fail to load config when it contains key starts with 
* fix: wrong lammps template_vars

## v0.12.0
* feat: support `dpff` training mode
* improvement: strict check of workflow configuration
* fix: wrong specorder of `fep-pka` mode
* breaking: `mode` move to general config section
* breaking: atom id of `DELETE_ATOMS` of `fep` mode now start from 1 instead of 0 (the same as lammps)
* breaking: require `deepmd-kit` when use `dpff` mode

## v0.11.3
* fix: incomplete lammps template_vars

## v0.11.2
* fix: pickle issue

## v0.11.1
* fix: typo

## v0.11.0
* feat: cll-training-workflow support `fep-redox` mode, details [here](./doc/manual/cll-workflow.md#train-mlp-model-for-fep-based-redox-potential-calculation
)
* feat: dpdata support `set_fparam`

## v0.10.16
* fix: ensure no duplicate dataset in deepmd input

## v0.10.15
* fix: lammps vars should also be template vars

## v0.10.14
* improvement: plumed config support lammps template vars, e.g `$$TEMP`

## v0.10.13
* feat: batch toolkit

## v0.10.12
* fix: openpbs workdir

## v0.10.11
* feat: support pbs job scheduler

## v0.10.10
* fix: dpdata filter

## v0.10.9
* refactor: aos-analysis options

## v0.10.8
* fix: dp train should support restart

## v0.10.7
* chore: update ai2cat default value

## v0.10.6
* chore: update ai2cat workflow template

## v0.10.5
* feat: Amorphous Oxides Structure Analysis toolkit, for more detail please check [here](./doc/manual/aos-analysis.md)

## v0.10.4
* fix: scale factor of fep lammps template

## v0.10.3
* feat: support customize fix statement

## v0.10.2
* fix: formily schema

## v0.10.1
* fix: ai2cat lammps template

## v0.10.0
* feat: release of ai2cat notebook

## v0.9.17
* fix: missing files

## v0.9.16
* feat: interactive ui for generate lammps input

## v0.9.15
* feat: yaml tool

## v0.9.14
* feat: ai2cat notebook

## v0.9.13
* chore: upgrade jupyter-formily

## v0.9.12
* wip: ai2cat tool and notebook
* fix: asap log error

## v0.9.11
* feat: ase tool support `set_by_ref` and `limit` method
* feat: integrate `jupyter-formily`
* wip: ai2cat tool and notebook

## v0.9.10
* fix: deepmd data path

## v0.9.9
* feat: deepmd option to group data by formula
* fix: fep lammps template
* fix: copy dataset in deepmd

## v0.9.8
* fix: fep lammps template
* fix: cp2k potential/basic-set parser
* fix: lammps config validator

## v0.9.7
* fix: fep lammps template

## v0.9.6
* fix: squeue options

## v0.9.5
* feat: support `broadcast_vars` in explore step, for more detail please check [here](./example/config/cll-mlp-training/explore-lammps.yml)

## v0.9.4
* refactor: deprecate `type_order` for deepmd >= 2.2.4, more detail please check [here](https://github.com/deepmodeling/deepmd-kit/pull/2732)
* fix: HPC executor resubmit error

## v0.9.3
* fix: asap selector
* fix: doc typo

## v0.9.2
* fix: inject locals to eval

## v0.9.1
* fix: selector screening function

## v0.9.0
* improvement: avoid unnecessary job submission

## v0.8.10
* fix: plumed template

## v0.8.9
* fix: deepmd template

## v0.8.8
* fix: cache for py38

## v0.8.7
* feat: new ai2-cat commands

## v0.8.6
* fix: drop deepmd v1 config format support as it is not compatible with v2.2.5

## v0.8.5
* feat: cll-workflow support sort structure by energy, for more detail please check [here](./example/config/cll-mlp-training/selector-model-devi.yml)

## v0.8.4
* feat: cll-workflow support screening by energy, for more detail please check [here](./example/config/cll-mlp-training/selector-model-devi.yml)

## v0.8.3
* feat: cll-workflow deepmd support fixture_models, for more detail please check [here](./example/config/cll-mlp-training/train-deepmd.yml)
* feat: `ai2cat` support gen plumed input.

## v0.8.2
* feat: handle outlier data in deepmd training stage, for more detail please check [here](./example/config/cll-mlp-training/train-deepmd.yml)
* feat: dpdata tool support filter systems by lambda expression, for more detail please check [here](./doc/manual/dpdata.md)
* improvement: cll-workflow now allow user to change init_dataset in each iteration.

## v0.8.1
* feat: `ase` toolkit support save atoms as `cp2k-inc` file, which can be used as `@include coord_n_cell.inc` macro in CP2K input file.

## v0.8.0
* feat: `ai4cat` toolkit, for more detail please check [here](./doc/manual/ai4cat.md)

## v0.7.4
* improvement: sorted input files for dpdata and ase tools
  * You may need to quote the input file name or else the file expansion will be handled by shell. e.g. `ai2-kit ase tool read './path/to/*.xyz'`

## v0.7.3
* fix: ase tool read data

## v0.7.2
* improvement: ignore `squeue` error
* fix: upgrade `dpdata` to 0.2.16 to support CP2K 2023.1

## v0.7.1
* fix: condition of selecting new explore systems

## v0.7.0
* feat: update explore systems for each iteration，for more detail please check [1](./example/config/cll-mlp-training/selector-model-devi.yml), [2](./example/config/cll-mlp-training/workflow-common.yml)
  * This version breaks the compatibility of checkpoint file, you may remove the checkpoint file before running this version.

## v0.6.8
* improvement: allow user to ignore lammps error

## v0.6.7
* fix: itemgetter bug: https://stackoverflow.com/a/48988896/3099733

## v0.6.6
* **BREAKING CHANGE**: cp2k input template no longer support `dict`, system data should be loaded via `@include coord_n_cell.inc` macro. More detail [here](./example/config/cll-mlp-training/label-cp2k.yml)

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
