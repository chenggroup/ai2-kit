# ai2-cat Toolkit

```bash
ai2-cat  #  or ai2-kit feat cat
```
This toolkit is a collection of useful functions for dynamic catalysis researching. 

## Usage

### Generate CP2K Input Files

This toolkit can be used to generate CP2K input files 
for a given system automatically with some simple commands.

You can run the following command to see the help message:
```bash
ai2-cat build-config gen_cp2k_input --help
```

For example, suppose you have a system file named `AuCu.xyz`,
and the basic set and potential files you want to use is in `CP2K_DATA_DIR`,
then you can run the following command to generate CP2K input files for this system:
```bash
ai2-cat build-config load_system AuCu.xyz - gen_cp2k_input \
    --basic_set_file BASIS_MOLOPT --potential_file GTH_POTENTIALS \
    --accuracy high --style metal --out_dir ./cp2k_input
```

The above command will generate two files in the `./cp2k_input` folder:
* `cp2k.inp`: The CP2K input file.
* `coord_n_cell.inc`: The coordinate and cell sections.
