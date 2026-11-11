#!/usr/bin/env python3
"""Run one prepared DeepMD train/freeze/test with integrity and output checks."""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from prepare_deepmd import dataset_hashes, file_sha256


def verify_prepared(workdir):
    """Reject obsolete preparation and input/data changes before starting a job."""
    manifest = json.loads((workdir / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Preparation manifest is outdated; prepare into a new directory")
    if file_sha256(workdir / "input.json") != manifest.get("input_sha256"):
        raise ValueError("input.json changed after preparation; re-prepare into a new directory")
    config = json.loads((workdir / "input.json").read_text())
    all_paths, groups = [], []
    for group in ("train", "validation", "test"):
        records = manifest.get(group)
        if not records:
            raise ValueError(f"Missing prepared {group} systems")
        geometries = set()
        for record in records:
            path = Path(record["path"]).resolve(strict=True)
            if not record.get("files_sha256") or dataset_hashes(path) != record["files_sha256"]:
                raise ValueError(f"{group} dataset changed after preparation: {path}")
            all_paths.append(path)
            geometries.update(record["geometry_sha256"])
        groups.append(geometries)
    if len(all_paths) != len(set(all_paths)) or any(
        groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("Prepared train/validation/test splits overlap")
    training = config["training"]
    for group, field in (("train", "training_data"), ("validation", "validation_data")):
        expected = [r["path"] for r in manifest[group]]
        if training[field]["systems"] != expected:
            raise ValueError(f"Actual {group} paths do not match the preparation manifest")
        if training[field].get("set_prefix", "set") != "set":
            raise ValueError("This helper supports only the standard set.* dataset layout")
    if training["numb_steps"] != manifest["steps"] or config["model"]["type_map"] != manifest["type_map"]:
        raise ValueError("Training settings do not match the preparation manifest")
    return manifest, config


def local_artifact(workdir, name):
    path = (workdir / name).resolve()
    if workdir not in path.parents:
        raise ValueError(f"Output must remain inside the fresh work directory: {name}")
    return path


def verify_curve(path, requested_steps):
    if not path.is_file():
        raise RuntimeError(f"Missing learning curve: {path}")
    data = np.loadtxt(path, ndmin=2)
    if data.size == 0 or data.shape[1] < 2 or not np.all(np.isfinite(data)):
        raise RuntimeError("Learning curve is empty, malformed or contains nonfinite metrics")
    steps = data[:, 0]
    if np.any(steps < 0) or np.any(steps != np.floor(steps)) or np.any(np.diff(steps) < 0):
        raise RuntimeError("Learning curve has invalid or decreasing step indices")
    if int(steps[-1]) != requested_steps:
        raise RuntimeError(f"Training incomplete: last logged step {steps[-1]:g}, expected {requested_steps}")
    return {"last_step": int(steps[-1]), "rows": len(data), "sha256": file_sha256(path)}


def read_test_table(path, columns, rows, names):
    if not path.is_file():
        raise RuntimeError(f"Missing test output: {path.name}")
    headers = [line[1:].strip().split(":")[-1].split()
               for line in path.read_text().splitlines() if line.startswith("#")]
    if names not in headers:
        raise RuntimeError(f"Unsupported test column header in {path.name}; inspect the installed DeePMD format")
    data = np.loadtxt(path, ndmin=2)
    if data.shape != (rows, columns) or not np.all(np.isfinite(data)):
        raise RuntimeError(f"Invalid test output {path.name}: expected {rows} finite rows and {columns} columns")
    return data


def verify_test(workdir, prefix, record, requested_frames):
    frames = min(requested_frames, record["frames"])
    energy_file = workdir / (prefix + ".e.out")
    force_file = workdir / (prefix + ".f.out")
    energy = read_test_table(energy_file, 2, frames, ["data_e", "pred_e"])
    force = read_test_table(force_file, 6, frames * record["n_atoms"],
                            ["data_fx", "data_fy", "data_fz", "pred_fx", "pred_fy", "pred_fz"])
    # dp test loads and evaluates the exported model. Successful finite detail
    # arrays provide the native inference check; an exit code alone does not.
    return {"frames": frames, "force_components": frames * record["n_atoms"] * 3,
            "energy_rmse_eV_atom": float(np.sqrt(np.mean((energy[:, 1] - energy[:, 0]) ** 2)) / record["n_atoms"]),
            "force_rmse_eV_A": float(np.sqrt(np.mean((force[:, 3:] - force[:, :3]) ** 2))),
            "energy_sha256": file_sha256(energy_file), "force_sha256": file_sha256(force_file)}


def execute(args):
    workdir = Path(args.workdir).expanduser().resolve(strict=True)
    if args.test_frames < 1:
        raise ValueError("--test-frames must be >= 1")
    # A fresh prepared directory prevents stale outputs from passing verification
    # and avoids overwriting previous logs. Resume through the TESLA module.
    if args.execute and any(p.name not in {"input.json", "manifest.json"} for p in workdir.iterdir()):
        raise RuntimeError("Run directory is not fresh; preserve artifacts and prepare a new directory")
    summary = {"status": "FAILED", "start_utc": datetime.now(timezone.utc).isoformat(),
               "workdir": str(workdir), "backend": args.backend, "commands": [], "tests": []}
    try:
        manifest, config = verify_prepared(workdir)
        curve = local_artifact(workdir, config["training"].get("disp_file", "lcurve.out"))
        local_artifact(workdir, config["training"].get("save_ckpt", "model.ckpt"))
        model_path = workdir / ("frozen_model.pth" if args.backend == "pt" else "frozen_model.pb")
        commands = [
            ("training", ["dp", f"--{args.backend}", "train", "input.json"]),
            ("freeze", ["dp", f"--{args.backend}", "freeze", "-o", model_path.name]),
        ]
        for i, record in enumerate(manifest["test"]):
            commands.append((f"test-{i}", ["dp", "test", "-m", str(model_path), "-s", record["path"],
                                          "-n", str(min(args.test_frames, record["frames"])), "-d", f"test-{i}"]))
        if not args.execute:
            return {"status": "DRY_RUN", "integrity_checked": True, "workdir": str(workdir), "commands": commands}
        if shutil.which("dp") is None:
            raise RuntimeError("DeePMD-kit executable 'dp' is not installed on this compute host")
        summary.update({"type_map": manifest["type_map"], "validation": manifest["validation"],
                        "test": manifest["test"], "input_sha256": manifest["input_sha256"]})
        for label, argv in commands:
            verify_prepared(workdir)
            # The helper owns these output names; reject arrays left by any other stage.
            if label.startswith("test-") and list(workdir.glob(label + ".*.out")):
                raise RuntimeError(f"Stale output before {label}; use a new run directory")
            log_path = workdir / f"{label}.log"
            with log_path.open("x") as log:
                result = subprocess.run(argv, cwd=workdir, stdout=log, stderr=subprocess.STDOUT, check=False)
            summary["commands"].append({"name": label, "argv": argv, "exit_code": result.returncode, "log": str(log_path)})
            if result.returncode:
                raise RuntimeError(f"{label} failed with exit code {result.returncode}; see {log_path}")
            if label == "training":
                summary["learning_curve"] = verify_curve(curve, manifest["steps"])
            elif label == "freeze":
                if not model_path.is_file() or not model_path.stat().st_size:
                    raise RuntimeError("Freeze reported success but model file is missing or empty")
            else:
                index = int(label.split("-")[1])
                summary["tests"].append(verify_test(workdir, label, manifest["test"][index], args.test_frames))
        verify_prepared(workdir)
        summary.update({"status": "TRAINED_UNQUALIFIED", "model": str(model_path),
                        "model_bytes": model_path.stat().st_size, "model_sha256": file_sha256(model_path),
                        "note": "Requested training steps and finite native E/F test outputs verified. Scientific acceptance criteria remain separate."})
        return summary
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        if args.execute:
            summary["end_utc"] = datetime.now(timezone.utc).isoformat()
            (workdir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--backend", choices=("pt", "tf"), default="tf")
    parser.add_argument("--test-frames", type=int, default=100)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args), indent=2))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
