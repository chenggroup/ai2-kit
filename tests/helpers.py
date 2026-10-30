import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ai2_kit.main", *args],
        check=True,
        capture_output=True,
        text=True,
    )
