# Single-model helper contract

These optional helpers prepare and execute one energy/force DeePMD model. They are not a committee trainer, TESLA controller, restart engine or scientific qualification procedure. Use the selected example's native commands when its schema/physics lies outside this contract.

## Preparation and immutable data

`prepare_deepmd.py` supports ordinary fixed-type, labelled NPY systems: `coord.npy` and `force.npy` have shape `(frames, 3 * atoms)`, `energy.npy` is `(frames,)` or `(frames, 1)`, and periodic `box.npy` is `(frames, 9)`. Nonperiodic systems need the existing `nopbc` marker. Malformed layouts are rejected without modifying the original data. Mixed-type and extra-input models need a specialised preparation route.

Preparation records a schema-version-2 manifest with the rendered input hash, full dataset file hashes (including labels and auxiliary files), exact-geometry fingerprints, split paths and the effective learning-rate configuration. Use stable dataset snapshots. A model input or dataset change requires preparation into a new directory. Old manifests are not silently upgraded.

An explicit `--decay-steps` overrides the exponential schedule's value even when the input is ordinary JSON without placeholders. Unsupported schedule types are rejected for that override. Without the option, literal template settings are retained; an `@DECAY_STEPS` placeholder uses max(1, steps // 20).

## Execution checks

Run the helper in the compatible DeePMD environment and only on authorised resources. Dry-run performs input/data integrity checks but launches no engine and writes no status. Actual execution starts from a directory containing only `input.json` and `manifest.json`; previous logs, models and test files are preserved by refusing reuse. Resume a real TESLA run through its controller instead.

Before each stage and after testing, the runner checks the input hash, all data hashes, split paths and overlap. It rejects changed input files, changed/added/removed data files and inconsistent training/validation paths. Intentional changes are a new preparation, not an in-place bypass of these checks.

After training, the learning curve must contain finite numeric metrics, valid nondecreasing integer steps and a final logged step equal to the requested step count. Training failures prevent freezing and testing. After freezing, the model must be nonempty. Native `dp test` then loads and evaluates it: zero exit alone is insufficient; fresh finite energy and force detail arrays, recognised column headers and the requested frame/atom counts must also be present. The summary records step completion, hashes and per-system prediction errors. It reports `TRAINED_UNQUALIFIED`, not scientific acceptance.

The output parser supports the native `.e.out` columns `data_e pred_e` and `.f.out` columns `data_fx data_fy data_fz pred_fx pred_fy pred_fz`. Check the installed version before execution; an unfamiliar format fails explicitly rather than guessing its column meanings. These contracts were checked against the [DeePMD v2.2.9 test implementation](https://github.com/deepmodeling/deepmd-kit/blob/v2.2.9/deepmd/entrypoints/test.py) and archived output headers. Backend CLI compatibility remains an environment check; this is not a claim that every DeePMD version has the same training flags.

The helper is synchronous. Arrange the actual failure/stall monitor and finite execution budget as specified by the execution module before launching it. The integrity checks detect ordinary file drift at stage boundaries; they do not replace stable snapshots or establish statistical independence of chemically correlated test structures.

## Checks performed on this revision

The local regression suite uses small synthetic NPY data and a temporary fake `dp`. It covers positive execution, incomplete steps, missing/nonfinite outputs, engine failure, stale files, input/label drift, split inconsistency, decay overrides and array axes. It does not train or evaluate a real model. Run the selected real backend's small compatibility checks before a new production run.

### Real backend smoke check

On 2 November 2026, the repaired helper path also completed a TensorFlow CPU smoke run with DeePMD-kit 3.1.3: 4/2/2 water frames for fitting/validation/testing, 20 training steps, freezing, and native E/F testing. All stages exited 0; the last step, finite output counts and hashes were verified. GPU availability and optional JAX/NumPy import diagnostics were recorded, although TensorFlow CPU execution completed. This small functional check is separate from the historical water demo and does not establish accuracy, GPU compatibility or an entire active-learning run.
