# Pinned upstream validation patterns

Source snapshot: [`chenggroup/ai2-kit`](https://github.com/chenggroup/ai2-kit), commit [`76181f47cbc2e204c8977af1b1d0e1a229bfe022`](https://github.com/chenggroup/ai2-kit/tree/76181f47cbc2e204c8977af1b1d0e1a229bfe022).

The tesla-h2o `test.sh` gathers `iter-00?/new-dataset/*` and `dp-init-data/*`, then runs native `dp test` on `iter-005/deepmd/model-0/compress.pb`. Its data are accumulated fitting data, so this is an in-scope fit/error diagnostic, not a held-out generalization test. `plot.sh` parses the resulting test output and plots model-deviation statistics from iteration screening TSVs. Preserve both the E/F error distributions and AL good/candidate/poor counts; inspect plot input paths and output artifacts.

`prod.sh` is an example-specific template: it retains `TYPE_MAP=[Ag,O]`, `iter-020` model paths, and example temperature settings. Its `DATA_FILE` is a generic LAMMPS data path (`lammps-data/000.data`). For water, adapt the type map, model iteration and any target-specific temperatures, MD settings, modules and filesystem paths. Do not present the template as a verified water recipe or reuse its example values blindly.

## Historical water case summary

The [water case summary](water-case.md) records an earlier executed workflow; its full execution archive is not included in this skill. No external archive directory is required to use the guide. The historical run used a resource-specific controller rather than the preserved upstream `test.sh`, `plot.sh` and `prod.sh` as a universal entrypoint.

For the example's scoped validation, check the native E/F errors and their distributions, AL counts/deviations, and learning/parity plots. The historical run also evaluated 16 separately labelled frames excluded from fitting. This is supplemental prediction evidence, not a universal scientific-qualification criterion or proof of independent-trajectory sampling.

## Source file links

- [tesla-h2o/test.sh](https://github.com/chenggroup/ai2-kit/blob/76181f47cbc2e204c8977af1b1d0e1a229bfe022/example/supplement/tesla-h2o/test.sh)
- [tesla-h2o/plot.sh](https://github.com/chenggroup/ai2-kit/blob/76181f47cbc2e204c8977af1b1d0e1a229bfe022/example/supplement/tesla-h2o/plot.sh)
- [tesla-h2o/prod.sh](https://github.com/chenggroup/ai2-kit/blob/76181f47cbc2e204c8977af1b1d0e1a229bfe022/example/supplement/tesla-h2o/prod.sh)
