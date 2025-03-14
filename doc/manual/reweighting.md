# Reweighting with ai2-kit

## Introduction
TODO

## Example

Given we have the following data:
* `traj.lammpstrj`: a LAMMPS trajectory file
* `COLVAR`: a PLUMED COLVAR file, the COLVAR file should align with the trajectory file
* `dp-baseline.pb`: a DeepMD baseline model
* `dp-target.pb`: a DeepMD target model


### Sampling 2000 frames from the trajectory and COLVAR file

We don't need to use the full trajectory for reweighting, so here we will use `ai2-kit` to sampling 2000 frames from the trajectory.  As the beginning frame may not be the best frame to start with, so we will also skip the first 100000 frames.

Note that we need to ensure the frames in the COLVAR file are aligned with the trajectory file, so we need to use the same random seed for both sampling.


```bash
# MUST use same random seed to ensure alignment
ai2-kit tool frame read traj.lammpstrj --rp 'TIMESTEP' - slice 100000: - sample 2000 --method random --seed 10 - write 2000.lammpstrj 
ai2-kit tool ase read 2000.lammpstrj --specorder [Ag,O] - to_dpdata - write 2000-dpdata

ai2-kit tool frame read COLVAR --frame_size 1 --header_size 1 - slice 100000: - sample 2000 --method random --seed 10 - write 2000-colvar --keep_header
```

The reason of not using ase tool to sample is ase will take extra time to parse the trajectory file,
by using frame tool to sample and then use ase tool to convert the data to dpdata format will save time.

Here we use the same `slice` and `sample` parameters for both `traj.lammpstrj` and `COLVAR` to ensure they are aligned.

### Use baseline and target model to calculate the energy

Now we need to use both the baseline and target model to calculate the energy of the sampled frames.

```bash
ai2-kit tool dpdata read 2000-dpdata/* --nolabel - eval dp-baseline.pb - write 2000-baseline
ai2-kit tool dpdata read 2000-dpdata/* --nolabel - eval dp-target.pb - write 2000-target
```
Note that we use `--nolabel` to ignore label in the dpdata file.

### Calculate reweighting FES

Now that we have all the data we need, we can calculate the reweighting FES.

```bash
 ai2-kit algorithm reweighting \
   load_energy 2000-baseline/**/energy.npy --tag baseline - \
   load_energy 2000-target/**/energy.npy   --tag target - \
   load_colvar 2000-COLVAR - \
   reweighting --cv d1 --bias opes.bias --temp 800 --save_fig_to fes.png --save_json_to result.json
```

The above command will calculate the reweighting FES and save the FES to `fes.png` and reweighting result to `result.json`.
