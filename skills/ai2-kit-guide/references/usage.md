# First use and example requests

These prompts illustrate supported requests; they are not records of completed simulations.

## Install or open the skill

Use a coding agent that can read project files and, for execution, run terminal commands on your authorised compute resources. Download the complete `skills/ai2-kit-guide/` folder, preserving its subdirectories. The skill itself does not install scientific engines.

For Codex, copy or symlink that folder into the project's `.agents/skills/ai2-kit-guide/`, then invoke `$ai2-kit-guide`. For Claude Code, use the project's `.claude/skills/ai2-kit-guide/` location and invoke `/ai2-kit-guide`. See the [Codex skill documentation](https://learn.chatgpt.com/docs/build-skills) and [Claude Code skill documentation](https://code.claude.com/docs/en/skills) for current host-specific installation details. Other file-capable coding agents can be explicitly directed to read the folder's `SKILL.md` and follow its linked modules. This manual route does not imply automatic discovery by every agent.

`agents/openai.yaml` contains optional Codex display/invocation metadata. It is not an OpenAI API dependency and is not needed to execute the scientific scripts. Keep `SKILL.md`, its referenced documents and the helper scripts together.

| What you want | Minimum information to provide | What you should receive |
| --- | --- | --- |
| Choose a starting example | Scientific objective, system and available structures/data | Recommended example, missing inputs and adaptation steps |
| Prepare a workflow | Input paths, reference method, requested engines and output directory | Edited files, checks and a concrete next command; no calculations |
| Fit existing labelled data | Real E/F labels, units/species/reference provenance, split policy, trainer and compute/time limits | Model, prediction tests, provenance and completion status |
| Run active learning | Initial labels or authorised reference bootstrap, structures, scientific/MD settings, compatible engines, resources and iteration/label/time limits | Iteration records, final models, requested tests and stop reason |

Coordinates alone are not energy/force training labels. The agent can help find appropriate existing labels or prepare an authorised labelling route; it must not invent them.

## Environment preflight

Use the selected route only. Record the Python executable/version, relevant package versions, resolved trainer executable/backend, available hardware and actual scheduler/host access in `environment-report.json`. Compare the trainer's environment with the Python used for preparation; do not assume the shell's default Python contains the required packages. Verify data files are real rather than Git LFS pointers.

For guidance/preparation, limit checks to source inspection, imports/help and rendered-input validation; do not invoke example `run.sh` scripts that submit jobs. For authorised supervised execution, check the trainer's small train/export/test path. Only active learning additionally needs MD inference and reference-calculation/parser checks. Record commands, exit codes and relevant diagnostics; distinguish optional-backend diagnostics from actual failure in the selected backend. Use an existing compatible environment where possible rather than modifying a shared environment.

## Execution handoff

For a prepared project, give the exact next command, working directory, environment, input/output paths and what it will run. For completed or interrupted work, give the last verified stage/step, model and log locations, stop reason and restart procedure. The single-model helper refuses directories containing old run outputs; resume through a verified trainer checkpoint command, not by deleting integrity checks. Full TESLA resumes through its workflow state.

Before launching an actual job, establish the failure/stall monitor described in the execution module. A generated command is not an unattended service, and a submitted job is not a completed model.

## Examples and evaluations

The bundled [evaluations](../evals/README.md) contain runnable mock fixtures and prompts for checking agent behaviour without scientific jobs. They are optional developer/user examples, not automatically executed when the skill is loaded. The [historical water case](execution/water-case.md) is a summary; the full original archive is not included in this skill distribution.

## Start from a scientific goal

> I have CP2K-labelled AIMD for a liquid electrolyte and use LAMMPS. Which Ai2-Kit
> workflow should I start with, and what do I need to change? Guidance only.

Select the water TESLA route if it matches the intended model, explain the four
stages and identify composition, labels, reference-method, model and sampling
changes. Ask for missing chemistry and data details; do not modify files.

## Continue within the same skill

> Use that route. Prepare a copy of the workflow for my supplied structures and
> CP2K inputs in `electrolyte-tesla/`. Keep my scientific parameters, leave cluster
> details as explicit placeholders, and check the scripts without running jobs.

Load the internal TESLA module, inspect the chosen template and supplied files,
make the requested copy/edits, update the README and perform focused local checks.
Return the files and the remaining settings, without another user-facing skill call.

## Substitute engines directly

> Adapt this water TESLA example to MACE, OpenMM and VASP using the supplied model
> and DFT inputs. Include compatible screening and check the code without launching
> training, MD or reference jobs.

Enter the TESLA module directly; inspect the existing engine-substitution example,
input/data contracts, model-deviation definition and environment requirements.

## Identify a missing stage

> I have a water energy/force potential and trajectory. Can I calculate Raman spectra?

Explain the missing polarizability data/model and spectral-analysis stages. Do not
equate an energy/force trajectory with a completed spectroscopy workflow.

## Train and run with supplied resources

> Use ai2-kit-guide to train an MLP for water from [data path], using my authorized
> CPU and GPU resources. Save the models and execution records in [output path].
> Use the example's checks; keep the run within [iteration/label/runtime budget].

Enter the execution module, recover data/reference provenance and actual compute
configuration, adapt the closest example, and complete the agreed TESLA stages.
Check real results, preserve restart state, and deliver models and analysis. Ask
only about missing consequential inputs or resource limits. The historical water
case illustrates execution; it does not guarantee new-system accuracy or a
successful clean deployment on every host.

## Resume a partially completed run

> Resume this interrupted water TESLA run within its existing budget. Do not
> resubmit completed DFT calculations or duplicate running jobs.

Enter the execution module, reconcile job state and input/output hashes, then
continue from verified artifacts. Check failure and stall monitoring before
reporting an unattended continuation.
