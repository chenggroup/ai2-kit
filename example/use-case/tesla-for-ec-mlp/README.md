# TESLA for ec-MLP

## Introduction
[ec-MLP](https://github.com/chenggroup/ec-MLP) is a machine-learning interatomic potential developed by the AI4EC Lab for *in situ*, dynamic, and high-precision simulations of electrochemical interfaces, designed to overcome the long-standing limitations of conventional theoretical approaches. By integrating a Deep Wannier neural network to describe the dielectric response of electrolytes with the Siepmann–Sprik polarizable model for metal electrodes, ec-MLP resolves the fundamental challenge arising from the distinct dielectric mechanisms of electronic conductors and ionic conductors at interfaces. Published in *Physical Review Letters* (July 2025), ec-MLP accurately reproduces *ab initio*–level water chemisorption/desorption dynamics and the experimentally observed bell-shaped differential capacitance at the Pt(111)–KF aqueous interface, and for the first time theoretically reveals the spatial distribution of the dielectric constant of interfacial water. ec-MLP provides a powerful new tool for multiscale, high-accuracy studies of realistic and complex electrochemical interface processes.


This workflow provides an example of how to use `ai2-kit` to build an active learning workflow for training ec-MLP models.
The workflow is based on [TESLA](../tesla/) workflow, with modifications to accommodate the specific requirements of ec-MLP training.


## Getting Started
To run the workflow, you need to ensure your environment has Python 3 and [mokit](https://anaconda.org/channels/mokit/packages/mokit/overview). 

And then all you need to do is to run the following command:

```bash
./run.sh
```

To customize the workflow, you can:
* Modify configuration in `00-config` folder, which include template file of `DeepMD`, `LAMMPS`, `CP2K`, `Slurm`, etc.
* Modify script in `02-workdir` to customize the workflow steps.

Send PR if you have any good ideas to improve the workflow.