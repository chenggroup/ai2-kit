# ai2-kit v2 — Stage 1 Implementation Plan

## Objective

Stage 1 establishes the breaking-change baseline for ai2-kit v2 on the
`v2-breaking` branch. It removes the deprecated configuration-driven product
surface, makes the package lighter, upgrades the supported Python and package
set, and introduces a small, reproducible test suite for the example-driven
tools.

The v2 result will eventually be merged into `main`. A separate
`v1-legacy` branch will be created and maintained manually; creating and
maintaining that branch is outside this stage.

## Scope and decisions

- The supported Python range starts at 3.10 and targets Python 3.10–3.14.
- NumPy 2 support is required. `dscribe` is retained and upgraded where
  needed.
- Configuration-driven CLL and FEP workflows are removed. New work uses the
  example-driven workflow approach.
- Deprecated HPC/Batch and checkpoint functionality is removed.
- `AseTool.write_each_frame` remains as a compatibility alias for
  `write_frames`.
- `asaplib` and selector clustering are removed. Clustering will be
  redesigned in a later stage.
- The old catalysis implementation, its UI, resources, CLI registration,
  documentation, and examples are removed.
- Shared domain modules remain when they are still required by tools or
  retained features (for example, CP2K, DPLR, and LAMMPS helpers).
- HPC execution and DeepMD/TensorFlow model loading are outside local and CI
  testing for this stage.

## Work packages

### 1. Inventory and dependency graph

Before deleting code, map imports from the workflow entry points and from the
retained `tool` and `feat` packages. Classify each module as:

1. removed because it exists only for configuration-driven workflows;
2. retained as a shared data or file-format utility; or
3. retained feature code that needs an import or API adjustment.

Record the resulting classification in the implementation PR so that shared
domain code is not removed accidentally.

### 2. Remove configuration-driven workflows

- Delete the CLL and FEP workflow entry points and their CLI registration.
- Remove workflow-only domain models, contexts, updater/checkpoint plumbing,
  and any executor/resource code that has no remaining consumer.
- Remove the CLL/FEP YAML configurations, workflow-specific examples, and
  corresponding English and Chinese manuals.
- Remove obsolete workflow and HPC configuration fixtures from `tests/`.
- Re-run import searches to ensure no retained module references deleted
  workflow symbols.

### 3. Remove deprecated HPC, Batch, checkpoint, and catalysis surfaces

- Delete the deprecated HPC and Batch tools and their command registration.
- Delete the checkpoint implementation and references left by removed
  workflows.
- Keep `write_each_frame` and add/retain a focused compatibility test for its
  delegation to `write_frames`.
- Delete `ai2_kit/feat/catalysis`, `ai2_kit/res/catalysis`, the `ai2-cat` and
  `feat cat` registrations, the catalysis manual, and catalysis-only examples
  and resources.
- Remove documentation links and package entry points that would otherwise
  expose deleted commands.

### 4. Update dependencies and packaging

- Change the Poetry Python constraint to `>=3.10` with the chosen upper-bound
  policy documented in `pyproject.toml`.
- Upgrade NumPy and `dscribe`, and update all directly affected packages to
  versions that install on Python 3.10–3.14 and support NumPy 2.
- Remove `asaplib`, `jupyter-formily`, and other dependencies/extras that are
  no longer imported after the deletions.
- Keep dependencies needed by retained ASE, dpdata, spectrum, and other
  example-driven functionality.
- Regenerate and review `poetry.lock`; verify installation independently on
  each target Python version.

### 5. Add v2 GitHub Actions coverage

- Add or adjust a workflow that triggers on `push` to `v2-breaking` and on
  pull requests targeting `v2-breaking`.
- Use a Python 3.10–3.14 matrix.
- Install Poetry dependencies from `poetry.lock` and install the project in a
  reproducible way.
- Enable Git LFS checkout for test data.
- Run the lightweight unit, feature, and smoke suites as one required job.
- Do not add HPC jobs or DeepMD/TensorFlow model-loading jobs.
- Do not change `main` CI behavior in this stage.

### 6. Reorganize and clean tests

Move retained tests into a structure that mirrors the supported surface:

```text
tests/
  tool/
  domain/
  feat/
  smoke/
  data/
```

- Remove expired workflow/HPC tests and tests for deleted catalysis or
  checkpoint APIs.
- Keep lightweight parsing, conversion, formatting, and numerical tests.
- Ensure tests do not import DeepMD/TensorFlow model loaders or require an HPC
  scheduler.
- Keep test names and fixtures explicit enough for the CI output to identify
  the failing tool or feature.

### 7. Add ASE and dpdata smoke coverage

For both the CLI and Python APIs, add smoke tests that:

- read a small supported input;
- perform a representative conversion or transformation;
- write the result; and
- read the result back and validate frame count, element symbols, coordinates,
  and available energy/force fields.

The CLI tests should invoke the installed `ai2-kit tool ase` and
`ai2-kit tool dpdata` commands as subprocesses. The Python tests should call
the corresponding tool classes/functions directly.

### 8. Curate and manage test data

- Select small real-world samples from cp2kdata, dpdata, ASE, or equivalent
  upstream projects.
- Store source project, version/commit, license, and the exact trimming or
  conversion procedure in a text manifest next to the fixtures.
- Trim large trajectories and outputs to the minimum frames/sections needed
  for the assertions.
- Track binary fixtures with Git LFS. Aim for no individual file larger than
  1 MiB and no more than 10 MiB of new test data in total.
- Configure CI checkout to fetch the required LFS objects.

### 9. Update user-facing documentation

- Rewrite the workflow section of `README.md` around example-driven usage.
- State clearly that configuration-driven CLL/FEP, HPC/Batch, checkpoint, and
  old catalysis features are removed from v2.
- Tell users who need those legacy features to use the separately maintained
  `v1-legacy` branch once it is provided.
- Remove links to deleted manuals, examples, commands, and extras.
- Document the Python 3.10–3.14 support target and the lightweight local test
  boundary.

## Implementation order

1. Complete the import/dependency inventory and record the removal map.
2. Remove workflow, HPC/Batch, checkpoint, and catalysis code and references.
3. Update `pyproject.toml`, regenerate `poetry.lock`, and resolve Python/NumPy
   compatibility issues.
4. Reorganize existing tests and remove obsolete fixtures.
5. Add curated LFS fixtures and the CLI/Python ASE and dpdata smoke tests.
6. Add the `v2-breaking` GitHub Actions matrix and verify lockfile installs.
7. Update README and remaining documentation links.
8. Run the full lightweight suite on every target Python version and perform a
   final repository-wide search for removed commands, imports, and docs.

## Acceptance criteria

- No configuration-driven CLL/FEP, HPC/Batch, checkpoint, or old catalysis
  command, import, manual, example, or package extra remains in v2.
- `write_each_frame` still works as a compatibility alias.
- The package installs from the lockfile on Python 3.10, 3.11, 3.12, 3.13,
  and 3.14, subject to documented upstream availability constraints.
- NumPy 2 and the upgraded `dscribe` configuration pass the retained tests.
- `v2-breaking` push and pull-request Actions pass the lightweight matrix.
- ASE and dpdata CLI and Python smoke tests pass using only curated LFS data.
- No test requires HPC infrastructure or loads a DeepMD/TensorFlow model.
- Test data provenance and trimming instructions are committed, and the data
  size budget is respected.
- README guidance points legacy users to `v1-legacy` and no longer advertises
  removed v2 functionality.

## Explicit non-goals

- Creating or maintaining `v1-legacy`.
- Changing `main` branch CI or merging `v2-breaking` into `main`.
- Reimplementing selector clustering.
- Running HPC, scheduler, remote-executor, DeepMD, or TensorFlow model tests.
- Designing the next-generation workflow API beyond preserving the existing
  example-driven examples.
