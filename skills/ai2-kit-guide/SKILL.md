---
name: ai2-kit-guide
description: >-
  Guide users from a scientific objective to an Ai2-Kit example, explain required
  inputs and adaptations, and build, modify or execute TESLA workflows when requested.
  Use for Ai2-Kit onboarding, example selection, MLIP active learning, changes of
  chemistry or simulation engines, and TESLA script preparation, MLFF training, execution, recovery or troubleshooting.
  Also use when users ask which Ai2-Kit application supports FEP, electrochemical
  interfaces or spectroscopy, or whether an unfamiliar task is supported.
---

# Ai2-Kit guide

Provide one entry point for understanding capabilities, selecting an example, and
implementing an authorized adaptation, and executing a requested MLFF workflow.
TESLA development and execution are internal capabilities of this skill; users
need not invoke a second skill to continue.

For installation, minimum inputs and example requests, see [first use and examples](references/usage.md).

## Determine the requested action

- **Explain/select:** answer from the scientific goal, select an example, explain
  reusable stages and missing inputs. Do not edit files when asked only for guidance.
- **Build/adapt TESLA:** read [the TESLA development module](references/workflow-adaptation.md)
  and perform the requested edits and focused checks. If the user already specified
  a workflow and change, enter this module directly without repeating onboarding.
- **Inspect/troubleshoot:** inspect the affected workflow, logs and installed APIs.
  Use the TESLA module for relevant script fixes. Explain the evidence for each fix.
- **Execute/train/resume:** distinguish fitting existing labelled data from a full active-learning loop first. Single-model supervised fitting does not require MD, DFT or a committee. Read [the TESLA execution module](references/execution/README.md).
  Recover inputs, resource access and budgets, configure the selected example, and
  carry out only the selected route: supervised train/export/test, or the authorized
  training/exploration/screening/labelling cycle, with
  monitoring, recovery, final retraining and result delivery. Do not repeat an
  authorization already given. A guidance-only or files-only request does not
  authorize calculations. Workflow completion and scientific qualification have
  different criteria; follow the scope and monitoring rules in the module.

## Understand and route the task

Read [the example index](references/examples.md), then inspect the relevant README,
entry script and templates in the available checkout. Prefer its actual structure
over assumed names. Record the checkout revision and the example consulted; if only
a partial snapshot is available, say so. Use official documentation or actual package
source to resolve version-specific API questions.

Match the requested observable and data requirements before chemical similarity.
Recommend one primary example and at most one useful alternative. Describe whether
the task is directly demonstrated, an adaptation, or needs a new integration. Never
present a chemically similar example as proof of new-system accuracy.

Explain the shortest useful route:

1. What the selected workflow produces and which scientific engines it connects.
2. Which structure/data operations and stages can be reused.
3. What changes in composition, labels, scientific settings, code and execution setup.
4. The smallest next action, and only the missing information needed for that action.

Computational scientists select and validate the physical model and system-specific
parameters. Example settings are starting points. Keep committee agreement,
independent prediction errors, DFT convergence, sampling convergence and agreement
with experiment distinct. Do not infer an independent test set from its name.

For spectra, identify the required dipole/polarizability data, temporal sampling and
analysis stages. For electrochemical response, check charge/potential control rather
than recommending ec-MLP merely because a user mentions an electrolyte. If the
indexed examples do not implement the requested physics, identify the missing
components and possible reusable pieces without promising unchanged execution.

## Continue into implementation

Carry the user's goal, selected source files, available data, requested engine or
chemistry changes, known settings and unresolved choices into the TESLA module.
For a request to run or resume, continue into the execution module with the same context.
Do not ask the user to repeat information already supplied. Work on authorized,
independent edits while leaving scientifically consequential unknowns visible.

For FEP, ec-MLP and spectroscopy, guidance and source inspection are available;
the presence of an example is not a tested specialized builder. Do not force these
tasks into the single-state TESLA recipe. Describe any additional implementation
needed and proceed only within the requested scope.

## Report what actually happened

Separate **recommended**, **files prepared**, **syntax checked**, **locally tested**,
**submitted**, **running** and **scientifically validated**. Report changed files,
checks and unresolved inputs. Do not claim time saved or successful simulations from
code generation alone. See [usage examples](references/usage.md) for the supported
transition from guidance to implementation.

## Contributors

- **Weihong Xu** — Developed the original `build-tesla` skill and
  workflow-adaptation tools incorporated into this guide.

- **Xuanjie Chen** — Developed the MLFF training, execution and recovery
  module, and prepared the water-training demonstration.

- **Sheng Bi** — Defined the scope of the integrated guide and coordinated
  its integration, review and documentation.
