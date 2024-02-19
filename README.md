# ai<sup>2</sup>-kit

[![PyPI version](https://badge.fury.io/py/ai2-kit.svg)](https://badge.fury.io/py/ai2-kit)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ai2-kit)](https://pypi.org/project/ai2-kit/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ai2-kit)](https://pypi.org/project/ai2-kit/)


A toolkit featured _**a**rtificial **i**ntelligence × **a**b **i**nitio_ for computational chemistry research.

*Please be advised that `ai2-kit` is still under heavy development and you should expect things to change often. We encourage people to play and explore with `ai2-kit`, and stay tuned with us for more features to come.*


## Feature Highlights
* Collection of tools to facilitate the development of automated workflows for computational chemistry research.
* Utilities to execute and manage jobs in local or remote HPC job scheduler.
* Utilities to simplified automated workflows development with reusable components. 

## Installation

You can use the following command to install `ai2-kit`:

```bash
# for users who just use most common features
pip install ai2-kit

# for users who want to use all features
pip install ai2-kit[all]
```

If you want to run `ai2-kit` from source, you can run the following commands in the project folder:

```bash
pip install poetry
poetry install

./ai2-kit --help
```
Note that instead of running global `ai2-kit` command, you should run `./ai2-kit` to run the command from source on Linux/MacOS or `.\ai2-kit.ps1` on Windows.

## Manuals
### Featuring Tools
* [ai2-cat](doc/manual/ai2cat.md): A toolkit for dynamic catalysis researching.

### Workflows
* [CLL MLP Training Workflow](doc/manual/cll-workflow.md) ([zh](doc/manual/cll-workflow.zh.md))

### Domain Specific Tools
* [Proton Transfer Analysis Toolkit](doc/manual/proton-transfer.md)
* [Amorphous Oxides Structure Analysis Toolkit](doc/manual/aos-analysis.md)

### General Tools
* [Tips](doc/manual/tips.md): useful tips for using `ai2-kit`
* [Batch Toolkit](doc/manual/batch.md): a toolkit to generate batch scripts from files or directories
* [ASE Toolkit](doc/manual/ase.md): commands to process trajectory files with [ASE](https://wiki.fysik.dtu.dk/ase/)
* [dpdata Toolkit](doc/manual/dpdata.md): commands to process system data with [dpdata](https://github.com/deepmodeling/dpdata/)

### Notebooks
* [ai2cat](notebook/ai2cat.ipynb)

### Miscellaneous
* [HPC Executor Introduction](doc/manual//hpc-executor.md) ([zh](doc/manual/hpc-executor.zh.md)): a simple HPC executor for job submission and management
* [Checkpoint Mechanism](doc/manual/checkpoint.md)


## Acknowledgement
This project is inspired by and built upon the following projects:
* [dpgen](https://github.com/deepmodeling/dpgen/tree/master/dpgen): A concurrent learning platform for the generation of reliable deep learning based potential energy models.
* [ase](https://wiki.fysik.dtu.dk/ase/): Atomic Simulation Environment.