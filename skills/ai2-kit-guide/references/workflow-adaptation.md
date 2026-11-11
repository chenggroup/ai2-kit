# Workflow adaptation

Inspect the project, adapt stage scripts/templates, preserve data interfaces,
check the edits and document the remaining inputs. This is an internal module
of ai2-kit-guide, loaded only when workflow changes are requested.

## Inspect before modifying

Read the project's applicable instructions and README. Inspect `run.sh`, the
configuration directory and the actual workflow scripts. Examples use layouts such
as `00-config`, `01-workflow`, `20-workflow` or `02-workdir`; infer their purpose from
the files rather than the directory name. Do not scan large runtime outputs without
need. Preserve user edits and original labelled data; use the requested output copy
or working tree for new implementations.

Identify training, exploration, screening and reference-labelling stages and the
files connecting them. For a new chemistry, check species ordering, masses, cell/PBC,
charge/spin, reference method, labels, model inputs and sampling conditions. Keep
uncertain scientific choices visible rather than silently inheriting water settings.

## Adapt the requested stages

Keep naming and directory conventions consistent. Reuse existing operations where
their input/output contracts match; add a Python function or engine adapter where
needed. An engine substitution may change model formats, MD inputs, trajectory
formats, labels and job launchers together. Do not merge labels from different
reference methods without an explicit, scientifically justified policy.

Trace ambiguous CLI calls to installed code and signatures. Ai2-Kit's Fire-based
CLI exposes Python methods; useful modules include `ai2_kit.tool.ase`,
`ai2_kit.tool.dpdata`, `ai2_kit.tool.model_devi`, `oh_my_batch.combo` and
`oh_my_batch.batch`. Resolve submission-module names from the installed version.
The bundled [source inspector](../scripts/get_pym_src.py) can print a module:

```sh
python /path/to/ai2-kit-guide/scripts/get_pym_src.py ai2_kit.tool.model_devi
```

Provide environment/scheduler templates where machine details are unavailable.
Do not fabricate an executable path, a Slurm partition or a scientific parameter
and describe it as ready to run. Update the workflow README with entry points,
dependencies, changed stages, required inputs and the checks actually performed.

## Preserve labels and screening semantics

When writing labelled XYZ data, explicitly use extended XYZ (`extxyz`) and verify
energy, forces, cells and units survive a small round trip. If failed label files
are skipped, record which files and why; skipping parsing failures is not evidence
that unconverged reference calculations have been filtered scientifically.

For models that do not emit the required deviation file, evaluate the committee on
the same ordered frames and atoms, then supply a compatible adapter. The existing
[MACE adapter](model-devi.py) is a starting reference, not a universal implementation:

- Its per-atom score is the **maximum over models of the norm of the force difference
  from the committee mean**, followed by max/min/mean over atoms. Do not call this a
  variance or assume it matches another engine's statistical definition. Document
  the definition and units; reassess thresholds if the statistic changes.
- The file columns are `step`, `max_devi_v`, `min_devi_v`, `avg_devi_v`,
  `max_devi_f`, `min_devi_f`, `avg_devi_f`.
- The supplied adapter uses finite placeholders for unavailable virial columns.
  These are not physical predictions. Confirm that the actual downstream code
  selects only a force column before using them; they do not automatically make a
  consumer ignore virials. Do not silently apply virial-based filtering.
- Verify the parser accepts the header and values, the row/frame counts match,
  atom/type ordering and force units agree, and timestep/frame identifiers align.
  Confirm model/ASE APIs against the installed version before using the reference.
- Committee disagreement is an acquisition diagnostic, not calibrated uncertainty,
  independent prediction accuracy or proof of convergence.

## Completion markers and restarts

A marker is trustworthy only if the corresponding command succeeded and required
outputs passed appropriate checks. In particular, do not use an unconditional
`touch` after a command that may fail. A minimal shell pattern is:

```sh
set -euo pipefail
if [ ! -f stage.done ]; then
    engine_command && check_required_outputs && touch stage.done
fi
```

Replace `engine_command` and `check_required_outputs` with the real invocation and
stage-specific checks. Check that outputs belong to the current attempt: a stale
trajectory left by an earlier attempt can make an existence/size check pass even
if this invocation produced nothing. Use a fresh staging directory or preserve and
move stale generated outputs aside before launching, as appropriate to the restart
contract; do not remove original inputs or reference data. Test this stale-output
case with a mocked successful command that writes nothing. A marker from a previous input/model version does not prove
that a new configuration has completed. Document the chosen invalidation/restart
policy. Preserve scheduler-level recovery separately from workflow dependencies.

## Focused verification and delivery

Check shell syntax with `bash -n`; parse/compile changed Python without executing
scientific calculations. Verify template tokens, paths and stage interfaces. Where
useful, run a small CPU fixture for data conversion, adapter parsing or failure and
restart handling. Fixtures and mocked engine commands are not physical calculations.

For preparation-only requests, preparing code does not imply executing training,
DFT or MD. For an authorized run, continue to the [execution module](execution/README.md)
within the same skill rather than stopping after file preparation. Syntax success alone does not establish runtime correctness.
Report exact checks and untested dependencies, then identify the smallest next test.
For any separately authorized long-running job, follow project instructions for
first completed work, failure/stall monitoring and evidence of continued progress.
