# Usage evaluations

These optional evaluations check whether a coding agent follows the skill and produces working project changes. They are not potential-accuracy benchmarks and are not run automatically when the skill is loaded.

## Restart task: portable inputs and executable checks

`evals.json` supplies the task and expected outcome. Its `files` entries are relative to **the directory containing evals.json**, not an internal development workspace. The two files in `fixtures/restart/` are self-contained bash mocks; no LAMMPS, DFT, trainer, GPU or cluster is needed. The input launcher is deliberately faulty: it can create a completion marker after failure. Do not copy it into a real simulation project.

1. Start a fresh agent session with this skill and provide the prompt from `evals.json` plus its two fixture files. Ask the agent to save its edited `lmp-run.sh` in a new output directory. Do not provide earlier answers or a reference solution.
2. Inspect the generated shell script before running it. The checker executes supplied code in temporary directories with a fixed mock engine; it is a functional test, not a security sandbox.
3. From the installed skill directory, run:

```bash
python3 evals/check_restart.py /path/to/agent-output/lmp-run.sh
```

The checker resolves its fixtures relative to its own file, so it also works when invoked from another working directory. It reports six checks: valid completion, no repeated successful execution, engine failure, absent output, empty output and a stale completion marker. Exit status is zero only when all pass. It uses Python's standard library and bash.

The unchanged input fixture should fail these checks; that failure is intentional. Keep the agent's response, changed files, checker output, skill version, agent/model version and any human intervention if reporting an evaluation. A passing launcher does not prove successful scientific execution or broad skill performance.

## Routing scenarios

`execution-routing.json` contains guidance, preparation, supervised training, active-learning and recovery prompts with expected scope. These are manually assessed behavioural scenarios, not executed scientific results. In particular, supervised fitting must not acquire an MD/DFT/committee requirement. Compare requested actions with the agent's actual files, commands and results, not just its final claims.

## Verification of this evaluation package

During the 3 November 2026 packaging revision, the checker was exercised on the deliberately faulty fixture and on a small hand-written correct launcher. The faulty fixture was rejected and the correct launcher passed all six mock checks. This verifies the checker/fixture plumbing; it is not a new independent agent benchmark. No fresh model training, MD or DFT was run for these evaluation files.
