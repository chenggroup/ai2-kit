# Cross-host execution for the TESLA MVP

Use this procedure when the user supplies separate CPU and GPU resources. Resource aliases supplied by the user are not themselves configured endpoints. Resolve actual connections from the user's authorized project/SSH setup; never infer usernames, credentials, allocations, or scheduler availability from a name.

## Discover and assign roles

Inspect each authorized host with small read-only checks: architecture, memory, disk, scheduler commands, available modules/environments, the selected training/MD/labeling executables, and GPU visibility/driver. A host advertising an RTX 3090 does not prove the GPU is allocated to this user or that its software stack is compatible.

Typical assignment, subject to the measured environment:

| Role | Placement | Required check |
| --- | --- | --- |
| Orchestrator | A persistent authorized host or the active agent session | Can reach both compute endpoints and preserve project state |
| Committee training | GPU resource | Real train/export; sequential models if only one GPU is allocated |
| Exploration | Compatible GPU or CPU resource | The chosen MD engine can load the actual exported model and produce valid trajectory/deviation files |
| Labeling | CPU allocation | The chosen reference-engine single point converges and its E/F output parses correctly |
| Analysis | Any suitable authorized environment | Reads retrieved data and produces reproducible metrics and figures |

Use the cluster scheduler for cluster jobs; never run substantial calculations on a login node. On a standalone GPU host, inspect its existing usage and allocation rules before assigning jobs. Do not terminate unrelated processes or consume all visible devices simply because they are visible.

Record an actual host-specific launch method and a matching status method. Slurm jobs need job IDs plus terminal state/exit code checks, including accounting when a job disappears from the live queue. Direct jobs need a persistent wrapper with logs and an exit-status file. A PID disappearing is not evidence of success.

## Explicit artifact handoffs

Different machines may not share paths. Maintain a mapping of logical project paths to each host's absolute work directory. Prefer project-relative paths within copied jobs; render host-specific paths when necessary. Do not copy training input files containing source-host absolute dataset paths and assume it will run on the destination.

1. Synchronize initial train/validation data and adapted inputs to the training host. Keep the independent test split outside training globs.
2. After every required model is exported and checked, transfer the complete committee to the exploration host if different.
3. Transfer selected structures, cell/type metadata, and DFT inputs to the labeling host.
4. After labeling success and parsing, transfer the validated new dataset back to the training host.
5. After final training and validation, retrieve model(s), reports, logs, plots, and their numeric data to the user's results location.

For each transfer, record source/destination, iteration, file list, sizes, and SHA-256 checksums. Stage incomplete transfers in a temporary directory, verify them, then expose them to the next stage. Do not synchronize a directory that is still being written. If direct host-to-host transfer is unavailable, relay through an authorized orchestration workspace; do not assume both remote hosts can SSH to each other. Transfer only task-relevant data and never copy private keys into job directories.

## Persistent execution state

Create a project `workflow-state.json` with:
- Run ID, source revision, plan/budget, logical-to-host path map, and input hashes.
- Current iteration and stage: prepare/train/explore/screen/label/accumulate/final-train/validate/collect.
- For each submitted job: host, scheduler/job ID or wrapper status path, input hash, timestamps, state, exit code, log location, retry count, and expected outputs.
- Completed transfer manifests, dataset generation, model hashes, and stop reason.

Write updates atomically. On restart, reconcile recorded jobs with actual host state before submitting anything. Never launch duplicates because the chat lost its connection. Validate output existence/content and input identity before trusting a marker. Preserve failed outputs for diagnosis.

## Continue until a real outcome

Do not stop after setup or submission when the user requested the complete loop. Completion requires all stages and checks in the agreed scope. For an example-scoped task, use the native checks described in [upstream validation](upstream-validation.md). Extended stability, sampling and property checks are additional requirements only for an explicitly requested full scientific qualification; see [the execution module](README.md). A first completed loop does not establish completion of a multi-round request. Poll at sensible intervals, check logs and terminal outcomes, repair bounded failures, and advance dependent stages only after verification. Monitor from the active session or an explicitly established persistent workflow controller. Do not promise future chat notifications unless a supported follow-up mechanism has actually been configured.

If the agent session cannot remain active, ensure an authorized persistent controller really exists before claiming the workflow will continue. Otherwise report what is running, save exact resume commands/state, and explain that automatic continuation is not established. Do not leave dependent stages implied but unimplemented.

Stop on a declared convergence condition, the explicit resource limit, an unrecoverable failure, or missing required input/access. Report which occurred. An iteration cap or no-candidate result alone does not prove a scientifically qualified potential. Final labeling must be followed by final training if the delivered model is to include those labels.

Deliver a concise user-facing result: completed stages/iterations, final model location, independent error metrics, physical checks/figures, resource usage where measured, and any remaining limitation. A successful SSH connection, package install, or scheduler submission is never the final result.
