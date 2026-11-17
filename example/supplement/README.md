# Supplementary Examples

This directory contains supplementary example code referenced in the `ai2-kit` paper preprint: [arXiv:2607.00613](https://arxiv.org/abs/2607.00613).

These examples can be used as starting points for reproducing the workflows discussed in the manuscript. They also illustrate how application-specific strategies can be introduced by composing ai2-kit tools and workflow steps, without changing the framework's internal code.

### [tesla-h2o](./tesla-h2o/)
A TESLA workflow example for bulk water.

### Highlights
Without modifying the TESLA framework, ASE tools add two application-level sampling strategies: include high-deviation configurations in the labeling data and refresh the initial structures for subsequent exploration using low-deviation configurations.

```bash
[ $USE_BAD_CONFS -gt 0 ] && {
    ai2-kit tool ase read $SCREENING_DIR/poor.xyz - sample $USE_BAD_CONFS --method random - \
        write_frames $LABELING_DIR/data/bad-{i:03d}.inc --format cp2k-inc
}
```

```bash
[ $UPDATE_MD_CONFS -gt 0 ] && {
    rm -f $WORK_DIR/lammps-data/* || true
    ai2-kit tool ase read $SCREENING_DIR/good.xyz -\
        sample $UPDATE_MD_CONFS --method random - \
        write_frames $WORK_DIR/lammps-data/{i:03d}.data --format lammps-data --specorder "$TYPE_MAP"
}
```

### [tesla-auco](./tesla-auco/)
A TESLA workflow example for the Au-CO2 system.

### Highlights
`omb combo` expresses dependencies between input parameters: the LAMMPS and PLUMED templates share run parameters such as temperature, and the bias factor is computed from the temperature to keep the generated configurations consistent.

```bash
omb combo \
    add_files DATA_FILE "$WORK_DIR/lammps-data/*" --abs -\
    add_file_set DP_MODELS "$DP_DIR/model-*/compress.pb" --abs - \
    add_var TEMP $MD_TEMP  - \
    add_var STEPS $MD_STEPS - \
    add_var SAMPLE_FREQ $SAMPLE_FREQ - \
    add_randint SEED -n 10000 -a 0 -b 99999 --uniq - \
    set_broadcast SEED - \
    compute BIASFACTOR "25000 // TEMP" - \
    make_files $LMP_DIR/job-{TEMP}K-{i:03d}/lammps.in --template $CONFIG_DIR/lammps/lammps.in - \
    make_files $LMP_DIR/job-{TEMP}K-{i:03d}/plumed.in --template $CONFIG_DIR/lammps/plumed.in - \
    make_files $LMP_DIR/job-{TEMP}K-{i:03d}/run.sh    --template $CONFIG_DIR/lammps/run.sh --mode 755 - \
    done
```

### [tesla-pka](./tesla-pka/)
A TESLA workflow example for the FEP pKa calculation.

### Highlights
An application-level structure-preparation step removes specified atoms from explored final-state configurations to form deprotonated structures, then converts both initial- and final-state structures to CP2K include format for labeling.

```bash
ai2-kit tool ase read $SCREENING_DIR/candidate-ini.xyz - sample $MAX_LABEL  - \
    write_frames $LABELING_DIR/job-ini-{i:03d}/coord_n_cell.inc --format cp2k-inc

ai2-kit tool ase read $SCREENING_DIR/candidate-fin.xyz - sample $MAX_LABEL  - \
    delete_atoms $ATOMS_TO_REMOVE --start_id 1 -\
    write_frames $LABELING_DIR/job-fin-{i:03d}/coord_n_cell.inc --format cp2k-inc
```

### [tesla-redox](./tesla-redox/)
A TESLA workflow example for the FEP redox-potential calculation.

### Highlights
The example explores structures on a mixed potential-energy surface. The LAMMPS template references two model sets, scaled independently by `LAMBDA_NEU` and `LAMBDA_RED`; `omb combo` combines the model files and mixing parameters to generate job configurations.

```lammps
pair_style hybrid/overlay &
           deepmd @DP_MODELS_NEU out_freq ${SAMPLE_FREQ} out_file model_devi_neu.out &
           deepmd @DP_MODELS_RED out_freq ${SAMPLE_FREQ} out_file model_devi_red.out
pair_coeff * * deepmd 1 O H
pair_coeff * * deepmd 2 O H

fix sampling_PES all adapt 0 &
    pair deepmd:1 scale * * v_LAMBDA_NEU &
    pair deepmd:2 scale * * v_LAMBDA_RED
```

```bash
omb combo \
    add_files DATA_FILE "$LAMMPS_INIT_DIR/*.data" --abs - \
    add_file_set DP_MODELS_NEU "$DP_DIR/model-neu-*/compress.pb" --abs - \
    add_file_set DP_MODELS_RED "$DP_DIR/model-red-*/compress.pb" --abs - \
    add_var TEMP $MD_TEMP - \
    add_var STEPS $MD_STEPS - \
    add_var SAMPLE_FREQ $SAMPLE_FREQ - \
    add_var LAMBDA_RED $LAMBDA_RED - \
    add_randint SEED -n 10000 -a 0 -b 99999 --uniq - \
    set_broadcast SEED LAMBDA_RED - \
    make_files $LMP_DIR/job-{TEMP}K-{LAMBDA_RED}-{i:03d}/lammps.in --template $CONFIG_DIR/lammps/lammps.in - \
    make_files $LMP_DIR/job-{TEMP}K-{LAMBDA_RED}-{i:03d}/run.sh    --template $CONFIG_DIR/lammps/run.sh --mode 755 - \
    done
```

### [ec-mlp](./ec-mlp/)
An ec-MLP workflow example for constant-potential molecular dynamics at the Cu(111)/water interface.

### Highlights
This application requires an additional DeepWannier model. The workflow snippet adds training-input generation, batch-job creation and submission, and failure recovery, showing how a new model component can be incorporated through application-level steps.

```bash
[ -f $DW_DIR/dw-setup.done ] && echo "skip dw setup" || {
    omb combo \
        add_file_set DP_DATASET "$INIT_DATASET_DIR/*" --format json-item --abs - \
        add_var STEPS $TRAINING_STEPS - \
        add_var DECAY $DECAY - \
        add_randint SEED -n 1 -a 0 -b 999999 --uniq - \
        make_files $DW_DIR/dw-model-{i}/dw.json --template $CONFIG_DIR/dw/dw.json - \
        make_files $DW_DIR/dw-model-{i}/dw.sh   --template $CONFIG_DIR/dw/dw.sh --mode 755 - \
        done

    omb batch \
        add_work_dirs "$DW_DIR/dw-model-*" - \
        add_header_files $CONFIG_DIR/dw/slurm-header.sh - \
        add_cmds "bash ./dw.sh" - \
        make $DW_DIR/dw-train-{i}.slurm  --concurrency 1

    touch "$DW_DIR/dw-setup.done"
}
cd $DW_DIR && omb job slurm submit "$DW_DIR/dw-train*.slurm" --max_tries 2 --wait --recovery $DW_DIR/slurm-recovery.json
```

### [spectra](./spectra/)
A spectroscopy workflow example for the air--water interface.

### Highlights
The example illustrates ai2-kit's batch tools beyond the main exploration-and-labeling loop: run CP2K recalculations for the x, y, and z polarization directions, submit the jobs, then collect the vibrational data and generate a dataset.

```bash
omb batch \
    add_work_dirs ./workdir/run-01/iters-0*/label-cp2k/tasks/0* - \
    add_header_files ./header.sh - \
    add_cmds "mpirun cp2k.popt -i ./workdir/recalculate/input.inp &> input.out" - \
    add_cmds "mpirun cp2k.popt -i ./workdir/recalculate/polar_x.inp &> polar_x.out" - \
    add_cmds "mpirun cp2k.popt -i ./workdir/recalculate/polar_y.inp &> polar_y.out" - \
    add_cmds "mpirun cp2k.popt -i ./workdir/recalculate/polar_z.inp &> polar_z.out" - \
    make ./cp2k-{i}.slurm --concurrency 20

omb job slurm submit "./cp2k-*.slurm" --max_tries 2 --wait --recovery slurm-recovery.json

ai2-kit tool dpdata \
    read ./workdir/run-01/iters-0*/label-cp2k/tasks/0* --fmt cp2k/viber \
          --lumped_dict '{O:4}' --cutoff 1.2 --output_file input.out --mode both \
          --wannier wannier.xyz --wannier_x wannier_x.xyz --wannier_y wannier_y.xyz --wannier_z wannier_z.xyz -\
    write ./dataset --v2 --sel_symbol="[O]"
```

### [skill-demo](./skill-demo/)
An AI-agent-generated TESLA workflow example built with the `build-tesla` skill.


## Notes

* The contents of this directory are intended for workflow reproduction.
* Raw or original datasets associated with the manuscript will be released separately.
* If you have questions about these examples, please start a discussion in GitHub Discussions.
