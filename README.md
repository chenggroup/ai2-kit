[![PyPI version](https://badge.fury.io/py/ai2-kit.svg)](https://badge.fury.io/py/ai2-kit)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ai2-kit)](https://pypi.org/project/ai2-kit/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ai2-kit)](https://pypi.org/project/ai2-kit/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.00613-b31b1b.svg)](https://arxiv.org/abs/2607.00613)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/chenggroup/ai2-kit)
![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/y/chenggroup/ai2-kit/main)


<p align="center"> <img src="doc/res/logo.png" alt="ai2-kit logo" width="240" /> </p>

`ai2-kit` (pronounced as "AI Squared Kit")  is a command line and python toolkit featured _**a**rtificial **i**ntelligence × **a**b **i**nitio_ for complex chemistry systems research.

> [!NOTE]
> Our `ai2-kit` paper has been submitted as a preprint: [arXiv:2607.00613](https://arxiv.org/abs/2607.00613).
> The paper-related examples are available under [example/supplement](./example/supplement/).


## Feature Highlights
* Collection of tools to facilitate the development of automated workflows for computational chemistry research.
* Use with [oh-my-batch](https://github.com/link89/oh-my-batch) to build your own workflow with shell script.
* Provide reusable skills under [./skills](./skills/) for building code related to `ai2-kit`.
  * [build-tesla](./skills/build-tesla): a skill for building TESLA workflows from existing examples.

### 2D API Design: MFI/O × Method Chaining

The command-line interface and Python API of `ai2-kit` follow the same 2D
interface design, formed by two orthogonal conventions:
Multi-files-in-and-out (MFI/O) along the file dimension, and method chaining
along the operation dimension.

- **File dimension: Multi-files-in-and-out (MFI/O).** A single API call can
  operate on a whole collection of files. Inputs are specified by wildcards,
  while outputs are specified by template variables. This covers both reading
  many files into one collection and writing many files or task directories
  from one collection, replacing explicit loops, path bookkeeping, and
  directory creation.
- **Operation dimension: method chaining.** A sequence of dependent steps can
  be composed into a single expression, using `-` as the separator in the CLI
  or chained calls in Python. A pipeline of reading, transforming, and writing
  data proceeds without intermediate variables or files.

**2D composition.** MFI/O extends the file dimension and method chaining
extends the operation dimension. The two dimensions are orthogonal and can be
extended independently, so the same processing pipeline can be applied to a
whole collection of inputs while also producing a whole collection of outputs.

For example, the paper's data-handling task---reading all LAMMPS trajectories
under `./lammps-*/`, randomly sampling 100 structures, and writing them as
individual VASP POSCAR files---can be expressed equivalently through both
interfaces.

**Python API**

```python
from ai2_kit.tool.ase import AseTool

(
    AseTool()
    .read("./lammps-*/traj.lammpstrj", format="lammps-dump-text")
    .sample(100, method="random")
    .write_frames("./vasp-{i:04d}/POSCAR", format="vasp")
)
```

**CLI**

```bash
ai2-kit tool ase read "./lammps-*/traj.lammpstrj" --format lammps-dump-text \
    - sample 100 --method random \
    - write_frames "./vasp-{i:04d}/POSCAR" --format vasp
```

## Environment Requirements

`ai2-kit` is developed and tested on Linux systems, and it should work on macOS as well.
For Windows, most of the `ai2-kit` features are expected to work. However, some third-party libraries (such as *dscribe*) may not function properly on Windows. 
In such cases, it is recommended to use `ai2-kit` through Windows Subsystem for Linux (WSL).

If you are using the latest version of Python, some third-party libraries may not yet provide pre-built binary releases, 
which can cause `pip install` to fail. It is therefore suggested to use `ai2-kit` with Python 3.10–3.12.
We strongly recommend creating a dedicated Conda environment to avoid unexpected issues caused by incompatible package versions.

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

## Usage
All `ai2-kit` commands can be accessed via the command line interface `ai2-kit`.
You can run the following command to see the available commands and sub-commands:

```bash
ai2-kit
# e.g.
ai2-kit tool
ai2-kit tool dpdata

```
For detailed usage of each command, you can run the following command to view the help information:

```bash
ai2-kit <sub-command> --help
# e.g.
ai2-kit tool dpdata read --help
```

## Manuals

### General Tools
* [ASE Toolkit](doc/manual/ase.md): commands to process trajectory files with [ASE](https://wiki.fysik.dtu.dk/ase/)
* [DPData Toolkit](doc/manual/dpdata.md): commands to process system data with [dpdata](https://github.com/deepmodeling/dpdata/)
* [Model Deviation Toolkit](doc/manual/model-deviation.md): a toolkit to filter structures by model deviation

### Workflows
#### Example Driven Workflows (Recommended)
These workflows are built with [oh-my-batch](https://github.com/link89/oh-my-batch) and example shell scripts, which can be easily adapted to your own research purpose.
It provides more flexibility and transparency to run and customize their own workflows.

* [TESLA](./example/use-case/tesla/): A customizable active learning workflow for training machine learning potentials.
* [TESLA PIMD](./example/use-case/tesla-pimd/): A customizable active learning workflow for training machine learning potentials with path integral molecular dynamics.
* [TESLA for ec-MLP](./example/use-case/tesla-for-ec-mlp/): A customizable active learning workflow for training machine learning potentials for electrolyte systems.


#### Config Driven Workflows
> [!WARNING]
> **Deprecated:** Config-driven workflows are deprecated and will be maintained only for backward compatibility.
> For new projects and future updates, please use the example-driven workflows above, which will be the primary approach going forward.

These workflows are driven by configuration files, which can be easily modified to fit your own research purpose.
* [CLL MLP Training Workflow](doc/manual/cll-workflow.md) ([zh](doc/manual/cll-workflow.zh.md))


### Featuring Tools
* [NMRNet](doc/manual/nmrnet.md): A toolkit for predict NMR with deep learning network.
* [Proton Transfer Analysis Toolkit](doc/manual/proton-transfer.md)
* [Amorphous Oxides Structure Analysis Toolkit](doc/manual/aos-analysis.md)
* [Re-weighting Toolkit](doc/manual/reweighting.md)
* [ai2-cat](doc/manual/ai2cat.md): A toolkit for dynamic catalysis researching.


### Online Apps and Notebooks
* [NMRPeak Online App](https://chenggroup.xmu.edu.cn/apps/nmrpeak): AI tools for NMR analysis.
* [Electrolyte Designer](https://ai4ec.ac.cn/apps/op-elyte-emulator): run electrolyte simulations with ease.
* [NMRNet Prediction](https://ai4ec.ac.cn/apps/nmrnet-predict): an online app to predict NMR chemical shifts with pre-trained NMRNet models.

## Tips
* [Tips](doc/manual/tips.md): useful tips for using `ai2-kit`
