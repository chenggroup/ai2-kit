# TESLA execution module

This internal module is entered through `ai2-kit-guide` when execution is requested. It extends the guide with resource configuration, real stage execution, recovery and result delivery; it is not a second skill invocation. Paths below are relative to this module unless stated otherwise.

## Select the execution route first

| Request | Run only | Required software |
| --- | --- | --- |
| Guidance or prepare files | Inspect, explain or generate/check files; do not launch scientific jobs | Only tools needed to inspect or prepare those files |
| Supervised training from existing labels | Validate/split labels, configure the trainer, train, export and test | Requested trainer and data tools; no MD/DFT engine or committee required |
| TESLA active learning | Train a committee, explore, screen, label, accumulate, retrain and test | Compatible trainer, exploration and reference engines plus data/screening tools |
| Resume | Inspect the existing route, artifacts and job state before continuing it | The same route's dependencies; do not start a second run |

Do not turn an existing-data training request into an active-learning job. The
single-model helpers provide a supervised route; their refusal to reuse an occupied
run directory is deliberate. They do not implement restart. Resume an interrupted
training job through its trainer's verified checkpoint/restart interface, preserving
logs and data hashes. Resume full TESLA through the selected workflow's state logic.

## Purpose

Adapt the closest official ai2-kit example to a user's requested MLFF workflow. Execute and verify the selected example's documented workflow and analyses by default; do not expand it into additional property calculations or independent reference campaigns. For an explicitly requested full scientific qualification, define and execute the broader system-specific plan in §5. The upstream example demonstrates reusable patterns, not a universal all-in-one runner.

Support different chemical systems and software combinations by adapting the closest official example. Honor requested trainers, MD engines, and DFT engines. DeePMD/LAMMPS/CP2K is one route, not a restriction or a universal default. MACE/OpenMM/VASP is another source-backed example. A named engine is not a guarantee of installed or interoperable support: inspect and test the actual interfaces. Do not switch DFT reference physics merely to fit available software.

Keep the implementation minimal: use official ai2-kit tools, oh-my-batch, and localized project-script changes. Do not create a new workflow framework or install every backend. Execute only the requested scope; configuration-only and supervised-only requests do not authorize an active-learning loop.

Read [upstream validation notes](upstream-validation.md) when adapting official tesla-h2o/skill-demo scripts. The bundled [water case summary](water-case.md) describes historical evidence; this package does not include its full execution archive and does not depend on it.

## 1. Resolve inputs and select an example

Inspect available data and project history for species, cells/PBC, units, target conditions, reference DFT method, software choices, and resource limits. A user may supply only a data path and resource names; recover routine details yourself and ask only for missing access, scientifically consequential inputs, or additional spending. Energy/force labels alone do not identify their DFT method; obtain provenance from original inputs/outputs where possible.

For active learning, read [example routing](example-routing.md), select the closest current official example, and pin its Git revision. For supervised fitting, select only the relevant trainer template and data preparation; do not initialise exploration or labelling stages. Inspect its README, run/setup/iteration scripts, templates, and relevant installed CLI/source. Copy it into a new project, preserving its directory conventions and the user's existing work. `../../scripts/execution/init_project.py` can copy a local example and record provenance; it does not configure or execute it. Verify required Git LFS assets contain real data.

When changing any engine or using a non-DeePMD route, read [engine adaptation](engine-adaptation.md). Preserve special physics in PIMD, charged, field-dependent, or other specialized routes rather than replacing them with ordinary TESLA settings. Do not pull FEP/spectroscopy/property-model branches into an ordinary MLFF request without need.

## 2. Discover and configure compute

Resolve the user's authorized CPU/GPU resources from project configuration and connection metadata. Check actual access, hardware/allocation, scheduler, disk, existing environments, modules, and required software. Resource names and GPU models are hints, not proof of access or compatibility. Do not scan unrelated hosts or expose credentials.

Prefer compatible existing environments; otherwise configure isolated user-owned environments or suitable official containers using current official instructions. Generate/adapt scheduler headers and launch scripts to real partitions/accounts/resources; use Slurm/LSF/OpenPBS through the installed omb interfaces where applicable, or a monitored direct runner on standalone hosts. Do not run substantial jobs on login nodes or change shared environments unnecessarily.

Use the [environment preflight](../usage.md#environment-preflight) for the selected route. For an authorised supervised run, check a small train/export/test path only. For active learning, additionally check MD inference against native E/F predictions and one reference calculation plus label parsing. Preparation-only requests stop at non-executing checks. Assign stages by actual compatibility and measured resources; committee members can train sequentially on one GPU. Record finite step/runtime limits for supervised fitting, and iteration/label/runtime limits for active learning. Ask only when critical limits/access are missing or new cost is required.

For separate hosts, read [cross-host execution](cross-host-execution.md): verify artifact transfers, rewrite host-specific paths, preserve job state, and avoid duplicate submissions. Save the actual environment and checks in `environment-report.json`, excluding credentials.

## 3. Prepare data and adapt the workflow

Verify real energy/force labels, species/order, finite values, units, cells/PBC, reference method, and convergence where available. Use ai2-kit ASE/dpdata conversion with the correct native parser. Explicitly use `--format extxyz` for labeled XYZ. Check converted E/F and frame counts; never silently skip conversion failures or substitute predicted/random labels for DFT.

Separate train/validation/test by independent trajectories or appropriately separated time blocks, recording identities and hashes. Held-out tests never enter fitting, tuning, or active-learning selection. Keep different compositions/atom counts in appropriate systems with consistent element mapping. If labels are absent, report the missing reference-data dependency. Bootstrap first-principles labels only when that calculation is within the user's authorised scope; an existing-data fitting request does not authorise DFT.

For supervised fitting, adapt species mapping, labelled datasets, trainer configuration and export/test settings. For active learning, additionally adapt masses, reference-engine inputs and data files, MD integration/sampling, and screening thresholds together. Use [TESLA stage contracts](tesla-execution.md) for stage mechanics and known pitfalls; its DeepMD/CP2K examples are route-specific, not mandatory engines.

Use `omb combo` for template/parameter combinations, `omb batch` for job packaging, and `omb job` for scheduler submission/wait/recovery where the selected example uses them. Inspect installed help/signatures rather than inventing a universal command shared by all backends. Validate shell syntax and rendered inputs before submitting.

## 4. Execute the selected route

### Supervised fitting

Validate the requested split and input manifests, train the requested model(s),
export a loadable model and run the requested prediction tests. Report the actual
last completed step, metrics, model path/hash and training/test split provenance.
Do not require exploration, new DFT labels, AL statistics or physical-property
production runs to complete this route. Read the [helper contract](helper-contract.md)
when using the bundled DeePMD helpers. Apply the monitoring rules below to actual
training; a launch command alone is not an unattended execution service.

### TESLA active learning

Once inputs, access, and authorized resource limits are clear, make routine stage decisions and execute the requested workflow independently within those limits. Do not pause for user confirmation after each Train/Explore/Screen/Label/Retrain stage. Ask only when a consequential scientific choice, missing access, increased budget, or external mutation falls outside the authorization. Keep user questions/status inquiries distinct from instructions to edit workflow artifacts: a question does not authorize changing project inputs. Persist state and recover from actual logs/artifacts after interruption.

1. Train a committee with independent seeds, normally following the selected example's model count. Verify loadable artifacts from every required member.
2. Explore using the chosen MD/search engine. Obtain nonempty, physically plausible trajectories and aligned uncertainty outputs.
3. Screen using ai2-kit and the validated uncertainty interface. Record thresholds, good/candidate/poor counts, failed reads, and discarded frames. No candidates is not automatically convergence; first exclude failed/empty exploration and excessive poor configurations.
4. Label selected configurations with the chosen first-principles engine using a consistent reference method. Check convergence, parse real E/F, and track successful/failed configurations.
5. Accumulate valid new labels and continue training/exploration within the declared convergence and resource limits. After final labeling, retrain/export on all accumulated training data before final testing and delivery.

Committee deviation requires multiple independently initialized models. Non-native deviation needs a verified adapter, not a fabricated column or a single-model estimate. Do not manipulate thresholds just to manufacture candidates. A separately requested single-model UQ method needs its own implemented and validated algorithm.

Monitor actual job outcomes and outputs; submission is not completion. Preserve commands, job IDs, exit codes, logs, seeds, versions, input hashes, transfers, and stage state. Validate `.done` markers against current inputs and outputs; invalidate dependent artifacts after changes. Diagnose failures and retry the same failure at most twice by default; preserve the precise blocker when repair is exhausted. Do not claim a workflow will continue after the session unless an actual persistent controller has been established.

## 5. Scientific validation and delivery

For supervised fitting, complete native prediction/error tests within the requested scope. For active learning, complete the selected example's native prediction/error comparison and AL statistics/plots. Read [model validation](model-validation.md) only when full scientific qualification is requested. For the active-learning example scope, report `upstream_scope_validation_complete: true` and `workflow_complete: true` when those requested stages actually pass. Keep this scope marker separate from scientific qualification: do not claim `QUALIFIED` without agreed, satisfied scientific criteria. Do not add extra property/reference computations as a completion gate unless the user asked for full scientific qualification. For a full-qualification request, before expensive production establish a system-specific plan: target states, independent E/F thresholds, reference sources, MD stability duration, sampling/convergence criteria, relevant properties, and budget. Propose source-backed choices from the task and seek agreement only on consequential unknowns. No fixed duration or error threshold qualifies every material.

For full scientific qualification only, perform independent E/F tests, equilibrated long-duration stability and structural checks, and relevant target-property validation. Include MSD/diffusion for liquids or diffusive systems; use appropriate phase/lattice/EOS, molecular/cluster, interface, or reaction metrics for other goals. Quantify uncertainty, convergence, and reference comparability. Short MD and low committee disagreement do not prove scientific accuracy. Within the authorized budget, address insufficient training coverage and revalidate; data used to guide fixes become development data, requiring fresh independent final tests.

Deliver only the artifacts belonging to the requested route. Give the exact next command (if preparation only), model/output paths, completion status and route-specific restart instructions. Deliver final model(s) with format/hash/type map/reference method and applicability range; adapted scripts/config/environment; data provenance/splits; stage logs and, for active learning, iteration statistics; validation metrics/plots with numeric data and analysis scripts; and `run_summary.json` with stop reason and artifact paths. Inspect generated figures. Return results to a location the user can access.

Record `workflow_complete` separately from `validation_complete`. For upstream-scoped work, use the explicit scope marker and do not infer `mvp_complete` or scientific qualification from it. For full-qualification work, report overall `mvp_complete: true` only when the requested workflow and scientific validation both pass. Use `PREPARED_NOT_RUN`, `BLOCKED_NEEDS_LABELS`, `BLOCKED_ENVIRONMENT`, `FAILED`, `SMOKE_TEST_ONLY`, `TRAINED_UNQUALIFIED`, or `QUALIFIED` accurately. `QUALIFIED` requires agreed scientific criteria; exhausted budgets, missing references, and unconverged properties remain explicit limitations, not waived gates. Skill/static/helper checks do not establish real training success.

## Monitoring and verified progress

Follow the project's monitoring instructions. After launch, wait for a completed unit of work or a failure; a startup banner or GPU allocation is not progress. Establish a failure/stall monitor during the launch session. Check completed-unit counts, log modification times and actual CPU activity together; measure rates from consecutive completed units. Snapshot live output before analysis. When copying code between hosts, transfer the complete relevant directory and verify content hashes before submission and after edits.

## Bundled helpers and invocation

The copy helper and `../../scripts/execution/plot_errors.py` are backend-independent utilities. `../../scripts/execution/prepare_deepmd.py` and `../../scripts/execution/run_deepmd.py` are optional **single-model DeepMD-only** helpers, not the TESLA orchestrator or an interface for other trainers. They require standard labeled NPY systems, a compatible JSON template, independent `--train/--validation/--test` inputs, Python/NumPy, and the supported DeePMD CLI. Read [the helper input/output contract](helper-contract.md) and inspect `--help` before use; exact geometry checks do not prove time independence. Use the original example's native trainer/export/test commands for other routes.

A short invocation is sufficient:

```text
Use $ai2-kit-guide. Data: [path]. CPU: [resource]; GPU: [resource].
Use [requested software, or choose a suitable official route].
Complete the requested upstream-scope active-learning workflow, native dp test/error plots, and result delivery. Add broader physical validation only when full scientific qualification is requested.
```
