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
pip install ai2-kit  

ai2-kit --help
```

If you want to run `ai2-kit` from source, you can run the following commands in the project folder:

```bash
pip install poetry
poetry install

./ai2-kit --help
```
Note that instead of running global `ai2-kit` command, you should run `./ai2-kit` to run the command from source.

## Manuals

```{tableofcontents}
```

## Acknowledgement
This project is inspired by and built upon the following projects:
* [dpgen](https://github.com/deepmodeling/dpgen/tree/master/dpgen): A concurrent learning platform for the generation of reliable deep learning based potential energy models.
* [ase](https://wiki.fysik.dtu.dk/ase/): Atomic Simulation Environment.
