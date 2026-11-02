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
- NumPy 2 support is required. `dscribe` is a default runtime dependency,
  with lightweight LMBTR calculation coverage. ASE is constrained to
  `>=3.26,<3.27` because DScribe 2.1.2 uses the positional `Atoms` constructor
  changed in ASE 3.27.
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

## Accepted execution adjustments

This plan reflects the final implementation decisions. The following original
requirements were dropped during execution and are not outstanding work:

- A separate import/dependency classification report in the implementation PR
  is not required; the scoped removal commits record the implementation.
- Keep test scripts in a flat `tests/` layout, with fixtures under
  `tests/data/`, rather than splitting scripts into tool/domain/feat/smoke
  subdirectories.
- Reuse the existing repository fixtures. Selecting new upstream samples,
  adding provenance/trimming manifests, migrating fixtures to LFS, and enforcing
  a new-data size budget are outside this stage.
- CLI smoke tests use the current interpreter with `python -m ai2_kit.main`.
  Separate validation of installed console-script entry points is not required.
- Update the shared test workflow for both `main` and `v2-breaking`; preserving
  the old `main` test matrix and installation behavior is not required.
- Documentation deployment modernization is outside this stage. Removal of
  obsolete documentation references remains in scope.

Historical changelog entries may retain references to removed v1 features.
Generic shell checkpoint helpers used by retained features are also retained;
the removed checkpoint surface is the old workflow checkpoint service.

## Work packages

### 1. Inventory and dependency graph

Before deleting code, map imports from the workflow entry points and from the
retained `tool` and `feat` packages. Classify each module as:

1. removed because it exists only for configuration-driven workflows;
2. retained as a shared data or file-format utility; or
3. retained feature code that needs an import or API adjustment.

Use import searches to guide scoped removal commits and preserve shared domain
code. A separate classification document is not a deliverable.

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

- Set the Poetry Python constraint to `>=3.10,<3.15`, matching the supported
  Python 3.10–3.14 range.
- Upgrade NumPy and `dscribe`, and update all directly affected packages to
  versions that install on Python 3.10–3.14 and support NumPy 2.
- Remove `asaplib`, `jupyter-formily`, and other dependencies/extras that are
  no longer imported after the deletions.
- Keep dependencies needed by retained ASE, dpdata, spectrum, and other
  example-driven functionality.
- Install `dscribe` by default, retain its NumPy 2 compatible version, and
  constrain ASE to the compatible range described above.
- Regenerate and review `poetry.lock`; use the CI matrix to verify installation
  on each target Python version.

### 5. Add v2 GitHub Actions coverage

- Adjust the shared workflow to trigger on pushes and pull requests targeting
  either `main` or `v2-breaking`.
- Use a Python 3.10–3.14 matrix.
- Install Poetry dependencies from `poetry.lock` and install the project in a
  reproducible way.
- Enable Git LFS checkout for `v2-breaking` pushes and pull requests; existing
  test fixtures do not require LFS.
- Run the lightweight suite, including ASE/dpdata smoke and LMBTR tests, in each
  matrix job using `poetry run python -m pytest tests
  --ignore=tests/test_md_spectra.py -k 'not DPFFDPData'`.
- Do not add HPC jobs or DeepMD/TensorFlow model-loading jobs.

### 6. Reorganize and clean tests

Keep the accepted flat script layout and group retained fixtures under data:

```text
tests/
  helpers.py
  test_*.py
  data/
    shared/
    cp2k_*/
```

- Remove expired workflow/HPC tests and tests for deleted catalysis or
  checkpoint APIs.
- Keep lightweight parsing, conversion, formatting, and numerical tests.
- Exclude the local-only spectrum dataset test and DeepMD-dependent DPFF tests
  from the lightweight suite. The selected suite must not load DeepMD/TensorFlow
  models or require an HPC scheduler.
- Exercise retained LMBTR generation on a small generated water structure,
  checking finite, normalized descriptors and rotation/translation invariance.
- Keep test names and fixtures explicit enough for the CI output to identify
  the failing tool or feature.

### 7. Add ASE and dpdata smoke coverage

For both the CLI and Python APIs, add smoke tests that:

- read a small supported input;
- perform a representative conversion or transformation;
- write the result; and
- read the result back and validate frame count, element symbols, coordinates,
  and available energy/force fields.

The CLI tests invoke `python -m ai2_kit.main tool ase` and
`python -m ai2_kit.main tool dpdata` as subprocesses using `sys.executable`
through `tests/helpers.py`. The Python tests call the corresponding tool
classes/functions directly.

### 8. Reuse and organize test data

- Move existing CP2K/DPLR fixtures into `tests/data/` and common ASE/dpdata
  inputs into `tests/data/shared/`.
- Reuse those fixtures for parsing, conversion, and roundtrip assertions.
- Generate the small LMBTR input in a temporary test directory.
- Keep existing fixture contents and Git storage unchanged. New upstream data
  curation, provenance manifests, trimming, LFS migration, and size gates are
  not Stage 1 deliverables.

### 9. Update user-facing documentation

- Rewrite the workflow section of `README.md` around example-driven usage.
- State clearly that configuration-driven CLL/FEP, HPC/Batch, checkpoint, and
  old catalysis features are removed from v2.
- Tell users who need those legacy features to use the separately maintained
  `v1-legacy` branch once it is provided.
- Remove links to deleted manuals, examples, commands, and extras from active
  documentation, including the Jupyter Book table of contents. Preserve
  historical changelog entries.
- Document the Python 3.10–3.14 support target and the lightweight local test
  boundary.

## Implementation order

1. Inspect imports and dependencies to identify shared code and removal scope.
2. Remove workflow, HPC/Batch, checkpoint, and catalysis code and references.
3. Update `pyproject.toml`, regenerate `poetry.lock`, and resolve Python/NumPy
   compatibility issues.
4. Keep test scripts flat, organize existing data, and remove obsolete fixtures.
5. Add CLI/Python ASE and dpdata smoke tests using existing fixtures, plus a
   lightweight LMBTR test using generated input.
6. Add the `v2-breaking` GitHub Actions matrix and verify lockfile installs.
7. Update README and remaining documentation links.
8. Run the lightweight suite locally and through the target Python CI matrix;
   search for removed commands, imports, examples, and active documentation
   references.

## Acceptance criteria

- No configuration-driven CLL/FEP, deprecated HPC/Batch, workflow checkpoint,
  or old catalysis command, import, manual, example, or package extra remains
  in the supported v2 surface. Active documentation has no references to those
  deleted resources; historical changelog entries are exempt.
- `write_each_frame` still works as a compatibility alias.
- The package installs from the lockfile on Python 3.10, 3.11, 3.12, 3.13,
  and 3.14, subject to documented upstream availability constraints.
- Default installation includes `dscribe` and its dependencies. NumPy 2 and
  the compatible ASE/DScribe configuration pass the retained LMBTR test.
- `v2-breaking` push and pull-request Actions pass the lightweight matrix.
- ASE and dpdata CLI and Python smoke tests pass using existing repository
  fixtures; CLI subprocesses use the current interpreter's module entry point.
- The selected lightweight suite requires no HPC infrastructure and loads no
  DeepMD/TensorFlow models.
- Test scripts remain flat and existing fixtures are organized under
  `tests/data/`; provenance manifests and LFS conversion are not required.
- README guidance points legacy users to `v1-legacy` and no longer advertises
  removed v2 functionality.

## Implementation and validation status

- Legacy example, fixture, and active documentation cleanup is complete in
  commit `1c1810d`, including `example/use-case/fep-hf`, the unused artifact
  fixture, and obsolete book entries.
- Default DScribe installation, the ASE compatibility constraint, the updated
  lockfile, and LMBTR smoke coverage are complete in commit `1ea4ff8`.
- Local validation after those changes passed on Python 3.13 with NumPy 2.5:
  25 tests passed and 2 DPFF tests were deselected using the lightweight suite
  command above. `poetry check --lock` and default `main,dev` installation also
  passed.
- The earlier branch CI matrix passed before these two follow-up commits.
  Remote matrix results for the follow-up commits have not been verified here.
- The dropped requirements listed above are accepted scope adjustments, not
  remaining implementation tasks.

## Explicit non-goals

- Creating or maintaining `v1-legacy`.
- Merging `v2-breaking` into `main`.
- Preserving the old `main` test workflow behavior or modernizing documentation
  deployment.
- A separate dependency classification report, test-script subdirectories,
  new fixture provenance/LFS requirements, or installed console-script tests.
- Reimplementing selector clustering.
- Running HPC, scheduler, remote-executor, DeepMD, or TensorFlow model tests.
- Designing the next-generation workflow API beyond preserving the existing
  example-driven examples.
