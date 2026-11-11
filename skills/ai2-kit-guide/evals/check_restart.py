#!/usr/bin/env python3
"""Check an agent-produced launcher using bundled mocks; no scientific jobs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def evaluate(launcher):
    launcher = Path(launcher).resolve(strict=True)
    fixture = Path(__file__).resolve().parent / 'fixtures/restart/engine.sh'
    checks = []
    with tempfile.TemporaryDirectory(prefix='ai2-restart-eval-') as scratch:
        root = Path(scratch)

        def case(name):
            p = root / name
            p.mkdir()
            shutil.copy2(launcher, p / 'lmp-run.sh')
            shutil.copy2(fixture, p / 'engine.sh')
            return p

        def invoke(p, behavior='success'):
            env = dict(os.environ, MOCK_ENGINE_BEHAVIOR=behavior)
            try:
                result = subprocess.run(['bash', 'lmp-run.sh'], cwd=p, env=env,
                                        capture_output=True, text=True, timeout=10)
                return result.returncode
            except subprocess.TimeoutExpired:
                return None

        def attempts(p):
            f = p / 'attempts.log'
            return len(f.read_text().splitlines()) if f.exists() else 0

        def record(name, passed):
            checks.append({'check': name, 'passed': bool(passed)})

        p = case('success')
        rc = invoke(p)
        record('success produces output and completion marker', rc == 0 and
               (p / 'lammps.done').is_file() and (p / 'trajectory.out').is_file() and
               (p / 'trajectory.out').stat().st_size > 0 and attempts(p) == 1)
        rc = invoke(p)
        record('completed job is not repeated', rc == 0 and attempts(p) == 1)

        for mode in ['fail', 'no_output', 'empty_output']:
            p = case(mode)
            rc = invoke(p, mode)
            record(mode + ' is not marked complete', rc is not None and rc != 0 and
                   not (p / 'lammps.done').exists() and attempts(p) == 1)

        p = case('stale_marker')
        (p / 'lammps.done').touch()
        rc = invoke(p, 'fail')
        record('missing output invalidates stale completion marker',
               rc is not None and rc != 0 and not (p / 'lammps.done').exists())

    return {'scope': 'mock launcher checks only; no scientific execution',
            'checks': checks, 'passed': all(c['passed'] for c in checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('launcher', type=Path, help='Agent-produced lmp-run.sh to check')
    args = parser.parse_args()
    result = evaluate(args.launcher)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
