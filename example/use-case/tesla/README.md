# TESLA Workflow
A bash script based workflow for training a machine learning potential automatically.

## Introduction
**T**rain-**E**xplore-**S**creen-**L**abel **A**ctive-learning (TESLA) workflow is a bash script based workflow for training a machine learning potential automatically. 

This workflow is inspired by [dpgen](https://github.com/deepmodeling/dpgen) and [ai2-kit](https://github.com/chenggroup/ai2-kit). 

It is a bash script built with `oh-my-batch` and `ai2-kit`, which makes it easy to customize. 
Developers can easily add their own steps to the workflow by modifying the bash script directly. 

## Getting Started
To run the workflow, you need to ensure your environment has Python 3. And then all you need to do is to run the following command:

```bash
./run.sh
```

To customize the workflow, you can:
* Modify configuration in `00-config` folder, which include template file of `DeepMD`, `LAMMPS`, `CP2K`, `Slurm`, etc.
* Modify training strategy by creating your own scripted by copying `01-workflow/iter-basic-dp-lammps-cp2k.sh` and modify it.
* Add more iterations by editing `run.sh`.

Send PR if you have any good ideas to improve the workflow.