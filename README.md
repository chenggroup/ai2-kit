# ai<sup>2</sup>-kit

[![PyPI version](https://badge.fury.io/py/ai2-kit.svg)](https://badge.fury.io/py/ai2-kit)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ai2-kit)](https://pypi.org/project/ai2-kit/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ai2-kit)](https://pypi.org/project/ai2-kit/)


A toolkit featured _**a**rtificial **i**ntelligence × **a**b **i**nitio_ for computational chemistry research.

*Please be advised that `ai2-kit` is still under heavy development and you should expect things to change often. We encourage people to play and explore with `ai2-kit`, and stay tuned with us for more features to come.*


## Feature Highlights
* Collection of tools to facilitate the development of automated workflows for computational chemistry research.
* Use with [oh-my-batch](https://github.com/link89/oh-my-batch) to build your own workflow with shell script.


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
# If you meet ConnectionError, you can try to set the max-workers to a smaller number, e.g
# poetry config installer.max-workers 4
poetry install
poetry run ai2-kit
```

## Manuals
### Featuring Tools
* [NMRNet](doc/manual/nmrnet.md): A toolkit for predict NMR with deep learning network.
* [ai2-cat](doc/manual/ai2cat.md): A toolkit for dynamic catalysis researching.

### Workflows
* [CLL MLP Training Workflow](doc/manual/cll-workflow.md) ([zh](doc/manual/cll-workflow.zh.md))

### Domain Specific Tools
* [Proton Transfer Analysis Toolkit](doc/manual/proton-transfer.md)
* [Amorphous Oxides Structure Analysis Toolkit](doc/manual/aos-analysis.md)
* [Reweighting Toolkit](doc/manual/reweighting.md)

### General Tools
* [Tips](doc/manual/tips.md): useful tips for using `ai2-kit`
* [Batch Toolkit](doc/manual/batch.md): a toolkit to generate batch scripts from files or directories
* [ASE Toolkit](doc/manual/ase.md): commands to process trajectory files with [ASE](https://wiki.fysik.dtu.dk/ase/)
* [DPData Toolkit](doc/manual/dpdata.md): commands to process system data with [dpdata](https://github.com/deepmodeling/dpdata/)
* [Model Deviation Toolkit](doc/manual/model-deviation.md): a toolkit to filter structures by model deviation

### Notebooks
* [ai2cat](notebook/ai2cat.ipynb)


## Acknowledgement
This project is inspired by and built upon the following projects:
* [dpgen](https://github.com/deepmodeling/dpgen/tree/master/dpgen): A concurrent learning platform for the generation of reliable deep learning based potential energy models.
* [ase](https://wiki.fysik.dtu.dk/ase/): Atomic Simulation Environment.
