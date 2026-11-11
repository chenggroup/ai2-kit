#!/usr/bin/env python3
"""Copy a local ai2-kit example and record provenance; never submit or install."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def initialize(repo, example, destination):
    repo = Path(repo).expanduser().resolve(strict=True)
    source = (repo / example).resolve(strict=True)
    relative = source.relative_to(repo)
    if relative.parts[0] != "example" or not source.is_dir():
        raise ValueError("Select a directory under the checkout's example/ tree")
    if not (source / "run.sh").is_file():
        raise ValueError("Selected example has no run.sh; inspect and copy manually")
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ValueError("Destination already exists; choose a new project directory")
    if destination == repo or repo in destination.parents:
        raise ValueError("Keep generated projects outside the source checkout")
    revision = git(repo, "rev-parse", "HEAD")
    dirty = git(repo, "status", "--porcelain", "--", relative.as_posix())
    files = {}
    lfs_pointers = []
    for file in sorted(source.rglob("*")):
        if "20-workdir" in file.relative_to(source).parts:
            continue
        if file.is_symlink():
            raise ValueError(f"Source contains a symlink; inspect and copy manually: {file}")
        if file.is_file():
            data = file.read_bytes()
            name = file.relative_to(source).as_posix()
            files[name] = hashlib.sha256(data).hexdigest()
            if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
                lfs_pointers.append(name)
    provenance = {
        "status": "COPIED_NOT_CONFIGURED_NOT_RUN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": "https://github.com/chenggroup/ai2-kit",
        "local_checkout": str(repo), "revision": revision,
        "example": relative.as_posix(), "source_changes": dirty,
        "source_sha256": files, "unresolved_lfs_pointers": lfs_pointers,
        "next": "Inspect and adapt species/data/physics/backend/host; do not run the copied entrypoint unchanged.",
    }
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("20-workdir", "__pycache__"))
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--example", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(initialize(args.repo, args.example, args.destination), indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
