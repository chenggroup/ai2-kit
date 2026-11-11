# Scientific model validation and plots

**Scope:** load this reference when the user requests full scientific qualification. For supervised fitting alone, report native prediction tests. For the selected active-learning example's validation scope, use [upstream validation](upstream-validation.md). Do not turn the additional property/reference calculations below into mandatory work for a smaller request.

Define target states, intended use and acceptance criteria before production. Match validation effort to the intended scientific claim. Literature results provide traceable context, not automatic thresholds for a new system. Save numerical data, analysis scripts, settings and plots.

## Using the paper's evidence

The [paper, arXiv:2607.00613v2](https://arxiv.org/abs/2607.00613v2), illustrates several validation types:

| Example | Evidence shown | How to use it in a new project |
| --- | --- | --- |
| Bulk-water potential, Figure 4 | AL classifications, MLMD/AIMD O–O RDF, energy/force comparisons and residuals | Separate training fit from independent prediction accuracy; choose species pairs appropriate to the new system |
| Cluster catalysis, Figure 5 | Potential comparisons and reaction/activation free energies | Cover reactive configurations; free-energy validation requires sampling and an appropriate budget |
| Spectroscopy, Figure 10 | DFT recalculation of 100 frames from an MLMD trajectory, property-model comparisons and spectra | Document selection and training exclusion before calling samples held out; dipoles/polarizabilities require specialised labels/models |
| Engine substitution, Figure 11 | Committee statistics, RDF and prediction comparisons | Check the adapted workflow's physical outputs, not only generated-script syntax |

Use the pinned example and actual installed software for executable commands. Obtain specialised reference properties from appropriate original studies or implementations; never invent reference curves.

## 1. Independent energy and force errors

- Final test frames must not enter fitting, hyperparameter choice or committee selection. Fresh DFT labels from later trajectories can support additional evaluation; record selection and frame identities. Report training/validation curves separately.
- Record reference DFT settings. Divide each frame's total energy by its own atom count before computing per-atom errors; a single fixed atom count is wrong for mixed compositions. Compute force errors over atomic Cartesian components, and report relevant species/state groups.
- Report RMSE, MAE, counts, units and outliers. If energy zero points are adjusted, retain raw errors and explain the adjustment; do not conceal errors by a fitted offset.
- Plot energy/force parity with equal axis ranges and the identity line, plus residual distributions. Label units, data split and counts. Downsampling a plot must not change full-data metrics.

Inspect native test headers and the installed version before converting outputs into `reference,prediction` CSV files. Do not guess column order or mix total and per-atom energies. Illustrative paths below must be replaced with real project paths:

```bash
python /path/to/ai2-kit-guide/scripts/execution/plot_errors.py \
  --csv /project/validation/energy_per_atom.csv \
  --quantity 'Energy per atom' --unit 'eV/atom' \
  --output /project/validation/energy
python /path/to/ai2-kit-guide/scripts/execution/plot_errors.py \
  --csv /project/validation/force_components.csv \
  --quantity 'Force component' --unit 'eV/angstrom' \
  --output /project/validation/force
```

The helper requires NumPy/Matplotlib and produces PNG/PDF plots and JSON metrics. It does not convert units, establish test independence or automatically qualify a model.

## 2. MD inference and stability

Check exported-model loading, species, masses, units and engine interfaces. Compare trainer-native and MD-engine E/F predictions on a few identical frames before short MD. Record initial configuration, ensemble, timestep, duration, temperature/pressure, seed and model hash.

Inspect NaNs, close contacts, unexpected dissociation, temperature/pressure/volume/density anomalies and drift. NVT total energy is affected by the thermostat and is not an NVE conservation test. When conservation matters, equilibrate then run NVE, normalise drift by atom count and time, and check timestep sensitivity to distinguish integration from potential errors.

Plot appropriate time series with equilibration/statistics intervals. A short non-crashing MD run is a smoke test, not evidence of long-term stability.

## 3. Structure and target properties

For liquids, compare appropriate partial RDFs under matching composition, temperature, density/pressure and statistical conditions. Do not fix O–O as the pair for all systems.

- Remove equilibration and report sampling interval, bins, distance range, PBC, volume and reference trajectory. Respect minimum-image/implementation limits; in a simple orthorhombic cell the distance limit is usually below half the shortest box length.
- Check same/different-species counting and density normalisation. Coordination numbers use the corresponding species density. Homogeneous 3D bulk normalisation is generally unsuitable for slabs/interfaces or isolated clusters.
- Estimate uncertainty from independent trajectories or time blocks, not correlated frames treated as independent samples. Save numerical curves and inspect plots. Without a reference, report the structure itself rather than claiming AIMD agreement.

Choose properties by intended use: crystal lattice/EOS/elastic or phonon properties; molecular/cluster geometry and relative energies; interface density/orientation/adsorption; reaction energies/barriers and reactive E/F. Do not run every possible test for every system.

For transport, compute MSD with correct PBC unwrapping, drift removal, dimensionality and particle definition. Establish a diffusive interval, fit diffusivity with uncertainty, and check fit-window/trajectory-length sensitivity. Use independent blocks or time origins with appropriate correlation handling. If the diffusive regime is not reached, report that diffusivity is unresolved rather than returning zero as a passing result. Check finite-size effects when relevant. Free energies require their own sampling and statistical convergence checks.

For active learning, also plot each round's training/new-label counts, good/candidate/poor fractions, deviation distributions and trust thresholds. State denominators and read/failed/discarded counts. Low committee deviation is not proof of low true error.

## Plan, decision and delivery

Before expensive full-qualification work, save a `validation-plan.json` containing target composition/phase/conditions, metrics/units, reference sources, tolerances, equilibration criteria, duration ranges, repeats/blocking, convergence criteria and budget. Propose task-specific criteria from available evidence; ask only about unresolved consequential choices. No universal duration or error threshold qualifies every material.

For intended MD use, assess stability over a duration justified relative to relevant relaxation/correlation times. Compare independent repeats or blocks; a thermostat must not mask structural failure or drift. Include transport only when relevant to the intended use. Use crystal, molecular, interface or reaction observables for the corresponding scientific goals.

Match reference state, ensemble, size and method. Distinguish experimental/DFT/nuclear-approximation differences from model errors. A stable trajectory without a suitable reference does not establish reference accuracy.

When a criterion fails, distinguish integration, sampling, reference mismatch and model error. Add training labels only within the authorised scope and budget. A test set used to guide repairs becomes development data; reserve fresh final evaluation data. Stop with explicit limitations if the budget is exhausted or a required reference is unavailable.

Deliver `validation-plan.json`, `validation-results.json` and a `validation/` directory containing metrics, numerical curves, scripts, plots, trajectory/reference locations and hashes, MD settings, splits and statistical intervals. Mark each criterion passed, failed, unconverged, missing reference or not applicable with reasons. Inspect generated plots. Report workflow completion separately from scientific qualification; `workflow_complete: true` is not `validation_complete: true` or `QUALIFIED`.

Check installed-version interfaces against [DeePMD testing](https://docs.deepmodeling.com/projects/deepmd/en/latest/test/test.html), [LAMMPS RDF](https://docs.lammps.org/compute_rdf.html) and [LAMMPS MSD](https://docs.lammps.org/compute_msd.html). RDF depends on neighbour lists and `special_bonds`; MSD needs correct image flags. The `average yes` option is not appropriate for ordinary diffusion analysis.
