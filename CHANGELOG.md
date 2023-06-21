# Changelogs

## v0.3.4
* feat: apply checkpoint to all workflows and job submission 
* feat: add checkpoint command line interface

## v0.3.3
* refactor: using asyncio to implement tasks and workflows

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