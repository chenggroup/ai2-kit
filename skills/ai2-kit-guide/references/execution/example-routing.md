# Official sources and example routing

Reference snapshot: [chenggroup/ai2-kit at 76181f47](https://github.com/chenggroup/ai2-kit/tree/76181f47cbc2e204c8977af1b1d0e1a229bfe022), inspected on 29 October 2026. Check the user's selected revision and installed CLI before execution; this index is not a promise that later versions are identical.

## Relationship to the paper and earlier skill

That snapshot provides the earlier workflow-adaptation skill under `skills/build-tesla/`; its English and Chinese entry files describe the same skill. Its development procedure is incorporated into [workflow adaptation](../workflow-adaptation.md). Installing a second skill is not required. The integrated guide additionally covers example selection, authorised execution, recovery and validation.

The [paper, arXiv:2607.00613v2](https://arxiv.org/abs/2607.00613v2), describes composable CLI/API operations, user-owned workflow scripts and the four TESLA stages. It does not establish configuration-free training across arbitrary chemistry. The bulk-water example's approximately 170 labelled configurations and the MACE/OpenMM/VASP example's 94 configurations are case results, not universal data requirements. Training-set comparisons do not establish independent prediction accuracy. Check that the data identified by the paper's availability statement are actually present, labelled and traceable.

The reference repository marks the older configuration-driven workflow route as deprecated. Prefer example-driven projects. Software names in a paper do not replace installed-version and interoperability checks.

## Select by goal

| Goal | Repository path | Checks before use |
| --- | --- | --- |
| Supervised fitting from existing labels | Trainer template from the closest example; optionally the bundled DeePMD helpers | Reference method, units, species, splits and trainer compatibility; no MD/DFT engine is required for fitting alone |
| General Deep Potential + LAMMPS + CP2K | `example/use-case/tesla/` | Templates contain Ag/O and cluster settings; adapt them rather than treating them as general defaults |
| PIMD exploration | `example/use-case/tesla-pimd/` | Beads, i-PI/MD interfaces, nuclear quantum treatment and labelling method |
| ec-MLP development | `example/use-case/tesla-for-ec-mlp/` | Charge/field handling, specialised model inputs and labels |
| Water Deep Potential | `example/supplement/tesla-h2o/` | Actual LFS AIMD labels, TensorFlow backend and complete templates; establish a separate test split rather than assuming the example test is independent |
| Au–CO2 | `example/supplement/tesla-auco/` | Cluster chemistry, reactive exploration and enhanced-sampling settings |
| MACE + OpenMM + VASP | `example/supplement/skill-demo/` | Export filenames/formats, OpenMM units, committee deviations and real POTCAR data |
| Electrochemical interfaces | `example/supplement/ec-mlp/` | Charge/potential control and response labels; this is not the ordinary single-state MLFF route |
| FEP pKa/redox | `example/supplement/tesla-pka/`, `example/supplement/tesla-redox/` | Distinguish training from consuming pretrained models and validating free energies |

Spectroscopy, Deep Wannier and NMR models are not ordinary interatomic energy/force models simply because they use `dp train`. Other trainers can be integrated through an appropriate example; the single-model DeePMD helpers do not run every MLIP framework. See the [application index](../examples.md) for spectroscopy and other application entrypoints.

## Obtain and copy an example

Inspect an existing local checkout first. If downloading, avoid fetching all large LFS assets by default. Replace the illustrative paths below with the actual destination:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/chenggroup/ai2-kit.git /path/to/ai2-kit
# Check out the selected revision, then retrieve only required LFS data:
git -C /path/to/ai2-kit lfs pull --include="example/supplement/tesla-h2o/00-config/aimd.xyz"
```

A file starting with `version https://git-lfs.github.com/spec/v1` is a pointer, not trajectory data. Inspect `run.sh` before invoking it: an example entrypoint may install packages and submit many jobs, so it is not a harmless preflight command.

## Interface sources

- [Repository README](https://github.com/chenggroup/ai2-kit/blob/main/README.md)
- [Supplementary examples](https://github.com/chenggroup/ai2-kit/tree/main/example/supplement)
- `ai2_kit/tool/ase.py`, `dpdata.py` and `model_devi.py`: actual Fire CLI methods.
- [DeePMD documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/): select the installed version when checking backend, train, freeze and test commands.
