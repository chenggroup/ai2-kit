[![PyPI version](https://badge.fury.io/py/ai2-kit.svg)](https://badge.fury.io/py/ai2-kit)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ai2-kit)](https://pypi.org/project/ai2-kit/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ai2-kit)](https://pypi.org/project/ai2-kit/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.00613-b31b1b.svg)](https://arxiv.org/abs/2607.00613)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/chenggroup/ai2-kit)
![GitHub commit activity (branch)](https://img.shields.io/github/commit-activity/y/chenggroup/ai2-kit/main)


<p align="center"> <img src="doc/res/logo.png" alt="ai2-kit logo" width="240" /> </p>

`ai2-kit` (pronounced as "AI Squared Kit")  is a command line and python toolkit featured _**a**rtificial **i**ntelligence × **a**b **i**nitio_ for complex chemistry systems research.

> [!NOTE]
> - Our paper has been submitted as a preprint: [arXiv:2607.00613](https://arxiv.org/abs/2607.00613).
> The paper-related examples are available under [example/supplement](./example/supplement/).
> - v2 introduces major changes to workflows and Python support. See
> [Appendix: Transitioning from v1 to v2](#appendix-transitioning-from-v1-to-v2)
> for details, including plans for maintaining v1.

## Feature Highlights
* Collection of tools to facilitate the development of automated workflows for computational chemistry research.
* Use with [oh-my-batch](https://github.com/link89/oh-my-batch) to build your own workflow with shell script.
* Provide reusable skills under [./skills](./skills/) for building code related to `ai2-kit`.
  * [build-tesla](./skills/build-tesla): a skill for building TESLA workflows from existing examples.

## 2D API Design Pattern

A key innovation of `ai2-kit` is its 2D API design pattern, shared by the
command-line interface and Python module. The two dimensions are:

- **File dimension:** A single API call can operate on a collection of files.
  Wildcards specify inputs, and template variables specify outputs, making
  batch processing as straightforward as working with a single file.
- **Operation dimension:** A sequence of dependent steps can be composed into
  a single expression. The CLI uses `-` to separate operations, while Python
  uses chained method calls. Reading, transforming, and writing data can be
  combined without intermediate variables or files.

By extending both the file and operation dimensions, `ai2-kit` provides
powerful tools through a concise interface. For example, the following
workflow merges trajectory files, samples structures, converts their format,
and writes each result to a separate file in a single expression:

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

**Command Line Interface**

```bash
ai2-kit tool ase read "./lammps-*/traj.lammpstrj" --format lammps-dump-text \
    - sample 100 --method random \
    - write_frames "./vasp-{i:04d}/POSCAR" --format vasp
```

## Environment Requirements

`ai2-kit` requires Python 3.10 or later. On Windows, use `ai2-kit` through
WSL2. We recommend installing it in a Conda environment.

## Installation

Install the core package with pip:

```bash
pip install ai2-kit
```

To install all optional features and their dependencies, including
`cp2kdata`, DeePMD-kit, and TensorFlow, etc, use:

```bash
pip install 'ai2-kit[all]'
```

Note that TensorFlow currently provides wheels for Python 3.10-3.13. The `tensorflow`
and `all` extras install it only on these Python versions.

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
Workflows are built with [oh-my-batch](https://github.com/link89/oh-my-batch) and example shell scripts, which can be easily adapted to your own research purpose.

* [TESLA](./example/use-case/tesla/): Customizable active learning workflows for training machine learning potentials, with variants for [path integral molecular dynamics](./example/use-case/tesla-pimd/) and [electrolyte systems (ec-MLP)](./example/use-case/tesla-for-ec-mlp/).

The [supplementary examples](./example/supplement/) accompanying the paper provide workflows for specific research applications:

* [TESLA for bulk water](./example/supplement/tesla-h2o/): Active learning for a bulk-water machine learning potential.
* [TESLA for Au-CO2](./example/supplement/tesla-auco/): Active learning for the Au-CO2 system.
* [TESLA for pKa](./example/supplement/tesla-pka/): Active learning for pKa calculations using free energy perturbation and thermodynamic integration (FEP-TI).
* [TESLA for redox potentials](./example/supplement/tesla-redox/): Active learning for redox-potential calculations using FEP-TI.
* [ec-MLP](./example/supplement/ec-mlp/): Constant-potential molecular dynamics at the Cu(111)/water interface.
* [Spectroscopy](./example/supplement/spectra/): IR, Raman, and sum-frequency generation spectra at the air-water interface using machine learning molecular dynamics and Deep Wannier models.
* [AI-agent-generated TESLA](./example/supplement/skill-demo/): A workflow built with the `build-tesla` skill, using MACE, OpenMM, and VASP.

### Featuring Tools
* [NMRNet](doc/manual/nmrnet.md): A toolkit for predict NMR with deep learning network.
* [Proton Transfer Analysis Toolkit](doc/manual/proton-transfer.md)
* [Amorphous Oxides Structure Analysis Toolkit](doc/manual/aos-analysis.md)
* [Re-weighting Toolkit](doc/manual/reweighting.md)


### Online Apps and Notebooks
* [NMRPeak Online App](https://chenggroup.xmu.edu.cn/apps/nmrpeak): AI tools for NMR analysis.
* [Electrolyte Designer](https://ai4ec.ac.cn/apps/op-elyte-emulator): run electrolyte simulations with ease.
* [NMRNet Prediction](https://ai4ec.ac.cn/apps/nmrnet-predict): an online app to predict NMR chemical shifts with pre-trained NMRNet models.

## Tips
* [Tips](doc/manual/tips.md): useful tips for using `ai2-kit`

## Developer Guide

### Install from Source

To install `ai2-kit` from source, run the following commands from the project root:

```bash
pip install uv
uv sync --locked
uv run ai2-kit
```

> [!NOTE]
> If you encounter connection errors, reduce uv's download concurrency:
> ```bash
> UV_CONCURRENT_DOWNLOADS=4 uv sync --locked
> ```

Jupyter Book is in the optional `doc` dependency group and is not installed by
default. To install the dependencies needed to build the documentation, run:

```bash
uv sync --locked --group doc
```

### Local Tests

Install the runtime and development dependencies from the lockfile, then run
the lightweight suite from the project root:

```bash
uv sync --locked --extra cp2kdata
uv run --no-sync python -m pytest -q tests --ignore=tests/test_md_spectra.py -k 'not DPFFDPData'
```

### Build Packages

```bash
uv build
```

## Appendix: Transitioning from v1 to v2

v2 makes several significant changes to how `ai2-kit` workflows are built and
maintained:

- **Example-driven workflows:** v2 replaces v1's configuration-driven workflow
  system with workflows built from runnable examples. This approach supports a
  wider range of functionality, makes workflows easier to reuse and adapt, and
  makes it easier to collaborate with AI coding agents.
- **Python 3.10 or later:** v2 drops support for Python 3.9 and earlier. This
  resolves compatibility issues with key dependencies and makes `ai2-kit` easier
  to install and use on newer Python versions.
- **Build tooling:** v2 replaces Poetry with uv for dependency management and
  package builds. Use `uv sync --locked` to install dependencies from `uv.lock`,
  `uv run` to run commands, and `uv build` to build distributions.
- **v1 maintenance:** The v1 codebase will move to the `v1-legacy` branch for
  continued maintenance.
