# Adapt software while preserving TESLA contracts

Read when replacing an engine or selecting a route other than the original example. Inspect official example code and the actual installed versions before designing commands. Do not assume every software combination has an existing executable template.

## Route selection

| User goal | Starting point | Adaptation to verify |
| --- | --- | --- |
| Deep Potential + LAMMPS + CP2K | `example/use-case/tesla/` or a suitable supplement | Native data/config, model format, plugin and reference inputs |
| MACE + OpenMM + VASP | `example/supplement/skill-demo/` | MACE training/export names, OpenMM interface, offline committee deviation, VASP reference files |
| Replace one engine | Closest working route above or another relevant official example | Every producer/consumer interface affected by the replacement |
| PIMD/ec-MLP or other specialized physics | Corresponding official route | Preserve its physical inputs, sampling and additional model/data requirements |

Respect explicitly requested engines. If no engine is specified, choose a source-backed route compatible with the reference data, target physics, and available resources; state why. Do not automatically choose CP2K or change the reference method because that route is familiar. Switching electronic-structure engines may change reference energies/forces even with nominally similar settings; establish compatibility or get an explicit consistent-labeling plan before merging labels.

## Stage contracts

1. **Data**: preserve species/order, cell/PBC, units, E/F, optional virials/stress and any required charge/spin/field/model inputs. Export the format accepted by the selected trainer. Retain explicit validation/test splits; an example's automatic random validation fraction is not a replacement for an independent test.
2. **Train/export**: use native executable and schema, actual save/export format and file names. Independently seed all committee members. Verify every model loads in its native inference API. Do not call `dp train`, `dp freeze` or `dp test` for another trainer or rename model extensions.
3. **Explore**: verify the MD calculator/plugin reads the model and preserves units, species mapping, atom order and cell conventions. Compare E/F on reference snapshots against native inference before MD. OpenMM/other engine units require explicit conversion checks. Use the requested ensemble/sampling, not copied default conditions.
4. **Uncertainty**: native DeePMD deviation may be consumed directly only after frame alignment checks. Otherwise evaluate the same trajectory with the complete committee through native/ASE calculators and write an ai2-kit-compatible force-deviation table.
5. **Label/accumulate**: generate correct structure and electronic inputs for CP2K, VASP, ABACUS or the selected engine. Verify actual required files and executables; never treat placeholder POTCAR/basis files as working inputs. Parse converged outputs with the correct parser, preserve atom mapping and reference method, and append only validated labels.
6. **Test/physics**: use the selected trainer's test/inference interface or a verified evaluator. Calculate errors against untouched real labels; retain the same scientific-validation requirements regardless of backend.

For the common force-committee convention, with forces shaped `(models, atoms, 3)`, compute per-component population standard deviation across models, then its vector norm per atom; take maximum/minimum/mean across atoms for force-deviation columns. Verify the convention against the chosen native implementation rather than calling it an ensemble variance. Require at least two independently initialized models. Check frame IDs/order, finite values, shape, zero deviation for identical predictions, and unit scaling on known inputs. If virial deviation is not computed, record it as unavailable and use only validated force columns for screening; compatibility placeholders are never physical virial results.

## Keep omb orchestration

Preserve the example's `omb combo` parameter/file expansion, `omb batch` packaging and `omb job` scheduler handling where applicable. Change training, MD, labeling templates and path/glob consumers together. Use scheduler-native monitoring for actual terminal outcomes. Reconcile local standalone hosts and remote clusters explicitly; changing engines does not remove cross-host transfer or recovery requirements.

Before scaling, execute one small real task across every changed boundary: trainer/export → MD inference → exploration/deviation → label parsing → next training input. Syntax checks alone are insufficient. Record the exact versions, commands, formats, conversion factors, checks and unresolved gaps in the project environment/run record. A failed compatibility check is a concrete repair task or blocker, not permission to claim arbitrary-backend support.

## Source references

- https://github.com/chenggroup/ai2-kit/tree/main/skills/build-tesla
- https://github.com/chenggroup/ai2-kit/tree/main/example/supplement/skill-demo
- Relevant installed ai2-kit ASE/dpdata/model_devi and omb CLI/source are authoritative for actual signatures.
