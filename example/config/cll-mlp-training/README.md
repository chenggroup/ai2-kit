# How to use config samples

## Overview
As `cll-mlp-training` workflow allows users to combine different configurations file to run the workflow, we provide some examples to show how to use them. You can just copy those files and update according to your needs.

## Example

For example, to train a DeepMD model with LAMMPS and CP2K, you can get started with the following samples:
* [executor.yml](./executor.yml)
* [artifact.yml](./artifact.yml)
* [workflow-common.yml](./workflow-common.yml)
* [train-deepmd.yml](./train-deepmd.yml)
* [explore-lammps.yml](./explore-lammps.yml)
* [label-cp2k.yml](./label-cp2k.yml)


You should copy the above files to your workspace and update them to your needs, then you can run the workflow with the following command:
```bash
ai2-kit workflow cll-mlp-training *.yml --executor hpc-cluster01 --path-prefix water/run-01 --checkpoint run-01.ckpt
```
