# ai<sup>2</sup>-kit

A toolkit featured ***a**rtificial **i**ntelligence × **a**b **i**nitio* for computational chemistry research.

*Please be advised that `ai2-kit` is still under heavy development and you should expect things to change often. We encourage people to play and explore with `ai2-kit`, and stay tuned with us for more features to come.*


## Feature Highlights
* A general purpose automated workflow that implements Closed-Loop Learning (CLL) pattern to train Machine Learning Potential (MLP) models.
* Featured tools for Electrochemistry research:
    * Automated FEP workflows to train MLP models and calculate redox potential, pKa, solvation, etc.
* Utilities to execute and manage jobs in local or remote HPC job scheduler.
* Utilities to simplified automated workflows development with reusable components. 

## Installation
```bash
# It requires Python >= 3.8
pip install ai2-kit  
```

## Use Cases

### Train MLP model with CLL workflow

```bash
ai2-kit cll-mlp train-mlp 
```

CCL-MLP workflow implements the Closed-Loop Learning pattern to train MLP models automatically. For each iteration, the workflow will train MLP models and use them to generate new training data for the next round, until the quality of MLP models meets preset criteria. Configurations of each iteration can be updated dynamically to further improve training efficiency.

![cll-mlp-diagram](./doc/img/cll-mlp-diagram.svg)

### Train MLP models for FEP simulation

```bash
ai2-kit ec fep train-mlp
```

`ec fep` is a dedicated workflow to train MLP models for FEP simulation. Unlike the general purpose `cll-mlp` workflow, `ec fep` workflow uses two different configurations to generate two different labeled structures to train MLP models respectively. And then use the two different models to run FEP simulation.

#### Citation
If you use `ec fep` workflow in your research, please cite it:
> Feng Wang and Jun Cheng, Automated Workflow for Computation of Redox Potentials, Acidity Constants and Solvation Free Energies Accelerated by Machine Learning. J. Chem. Phys, 2022. 157(2), 024103. DOI: https://doi.org/10.1063/5.0098330


## TODO
* Finalize configurations format and provide documents.
* Provide tools for data format transformation.
* Provide tools to run MD simulation and properties calculations.
