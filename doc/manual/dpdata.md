# Dpdata Toolkit

```bash
ai2-kit tool dpdata
```

This toolkit is a command line wrapper of [dpdata](https://github.com/deepmodeling/dpdata) to allow user to process DeepMD dataset via command line.

## Usage

```bash
ai2-kit tool dpdata # show all commands
ai2-kit tool dpdata to_ase -h  # show doc of specific command
```

This toolkit include the following commands:

| Command    | Description                                                                                                                 | Example                                                                                        | Reference                                   |
| ---------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------- |
| read       | Read dataset into memory. This command by itself is useless, you should chain other command after reading data into memory. | `ai2-kit tool dpdata read ./path/to/dataset --fmt deepmd/npy`                                  | Support wildcard, can be call multiple time |
| write      | Use MultiSystems to merge dataset and write to directory                                                                    | `ai2-kit tool dpdata read ./path/to/dataset --fmt deepmd/npy - write ./path/to/merged_dataset` |                                             |
| filter     | Use lambda expression to filter dataset by system data.                                                                     | See in `Example`                                                                               |                                             |
| set_fparam | add `fparam` to dataset, can be float or list of float                                                                      | See in `Example`                                                                               |                                             |
| slice      | use slice expression to process systems                                                                                     | see in `Example`                                                                               |                                             |
| sample     | sample data by different methods, current supported method are `even` and `random`                                          | see in `Example`                                                                               |                                             |
| eval       | use `deepmd DeepPot` to (re)label loaded data                                                                               | see in `Example`                                                                               |                                             |
| to_ase     | convert dpdata format to ase format and use [ase tool](./ase.md) to process                                                 | see in `Example`                                                                               |                                             |

Those commands are chainable and can be used to process trajectory in a pipeline fashion (separated by `-`). For more information, please refer to the following examples.

## Example

```bash
# read multiple dataset generated by training workflow by wildcard and merge them into a single dataset
# you can also call `read` multiple times to read multiple dataset from different directory
ai2-kit tool dpdata read ./workdir/iters-*/train-deepmd/new_dataset/* --fmt deepmd/npy - write ./merged_dataset  --fmt deepmd/npy

# You can also save data with hdf5 format
ai2-kit tool dpdata read ./workdir/iters-*/train-deepmd/new_dataset/* --fmt deepmd/npy - write ./merged.hdf5 --fmt deepmd/hdf5

# Use lambda expression to filter outlier data
ai2-kit tool dpdata read ./path/to/dataset --fmt deepmd/npy - filter "lambda x: x['forces'].max() < 10" - write ./path/to/filtered_dataset

# Set fparam when reading data
ai2-kit tool dpdata read ./path/to/dataset --fmt deepmd/npy --fparam [0,1] - write ./path/to/new_dataset

# (re)label data
ai2-kit tool dpdata read dp-h2o --nolabel - eval dp-frozen.pb - write new-dp-hwo

# Drop the first 10 frames and then sample 10 frames use random method, and save it as xyz format
ai2-kit tool dpdata read dp-h2o - slice 10: - sample 10 --method random - to_ase - write h2o.xyz
o

# convert cp2k data to the format that can be used by deepmd dplr module
# data used in v3
ai2-kit tool dpdata read ./path-to-cp2k-dir --fmt cp2k/dplr --cp2k_output="output" --wannier_file="wannier.xyz" --type_map="[O,H,K,F]" --sel_type="[0,2,3]" - write ./v3-dataset
# data used in v2
ai2-kit tool dpdata read ./path-to-cp2k-dir --fmt cp2k/dplr --cp2k_output="output" --wannier_file="wannier.xyz" --type_map="[O,H,K,F]" --sel_type="[0,2,3]" - write ./v2-dataset --v2 --sel_symbol="[O,K,F]"
# data with wannier spread (which works for both v2 and v3)
ai2-kit tool dpdata read ./path-to-cp2k-dir --fmt cp2k/dplr --cp2k_output="output" --wannier_file="wannier.xyz" --wannier_spread_file="wannier_spread.out" --type_map="[O,H,K,F]" --sel_type="[0,2,3]" - write ./v3-dataset-with-spread
# if wannier charge is not -8
ai2-kit tool dpdata read ./path-to-cp2k-dir --fmt cp2k/dplr --cp2k_output="output" --wannier_file="wannier.xyz" --type_map="[O,H,Li]" --sel_type="[0,2]" --model_charge_map="[-8,-2]" - write ./LiOH-dataset
```

## Note about data conversion between v2 and v3

The format of `atomic_dipole` and `atomic_polarizability` data used in DeepMD-kit v2 and v3 are different.
In v2, the required shape of the `atomic_*.npy` is `[n_frames, n_sel_atoms * n_dim]`, while in v3, the shape should be `[n_frames, n_atoms * n_dim]`. More details can be found in the [offcial doc](https://docs.deepmodeling.com/projects/deepmd/en/master/data/system.html). You can specify the version when generating the data:

```bash
# data used in v2
ai2-kit tool dpdata read ./path-to-cp2k-dir --fmt cp2k/dplr --cp2k_output="output" --wannier_file="wannier.xyz" --type_map="[O,H,K,F]" --sel_type="[0,2,3]" - write ./v2-dataset --v2 --sel_symbol="[O,K,F]"
# data used in v3
ai2-kit tool dpdata read ./path-to-cp2k-dir --fmt cp2k/dplr --cp2k_output="output" --wannier_file="wannier.xyz" --type_map="[O,H,K,F]" --sel_type="[0,2,3]" - write ./v3-dataset
```

You can also convert the data between v2 and v3 by using the following command:

```bash
from ai2_kit.domain.dplr import dplr_v2_to_v3, dplr_v3_to_v2

# change in place
dplr_v2_to_v3("dataset-v2", sel_symbol=["O", "K", "F"])
dplr_v3_to_v2("dataset-v3", sel_symbol=["O", "K", "F"])
```
