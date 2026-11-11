#!/usr/bin/env python3
"""Validate labeled DeepMD NPY systems and render one DeePMD model input."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_hashes(path: Path) -> dict:
    """Include labels, auxiliary arrays, type files and the presence of nopbc."""
    return {p.relative_to(path).as_posix(): file_sha256(p)
            for p in sorted(path.rglob("*")) if p.is_file()}


def inspect_system(raw_path: str) -> dict:
    path = Path(raw_path).expanduser().resolve(strict=True)
    before_hashes = dataset_hashes(path)
    types_file = path / "type.raw"
    map_file = path / "type_map.raw"
    if not types_file.is_file() or not map_file.is_file():
        raise ValueError(f"{path}: require type.raw and type_map.raw")
    type_map = map_file.read_text().split()
    atom_types = [int(x) for x in types_file.read_text().split()]
    if not atom_types or not type_map or any(x < 0 or x >= len(type_map) for x in atom_types):
        raise ValueError(f"{path}: invalid atom types/type map")

    if len(type_map) != len(set(type_map)):
        raise ValueError(f"{path}: duplicate element names")
    fingerprints = set()
    frames = 0
    for sample in sorted(path.glob("set.*")):
        if not sample.is_dir():
            continue
        arrays = {}
        if (sample / "real_atom_types.npy").exists():
            raise ValueError(f"{sample}: mixed-type systems need their native preparation route")
        for key in ("coord", "force", "energy"):
            file = sample / f"{key}.npy"
            if not file.is_file():
                raise ValueError(f"{sample}: missing labeled {key}.npy")
            arrays[key] = np.load(file, allow_pickle=False)
            if not np.all(np.isfinite(arrays[key])):
                raise ValueError(f"{file}: nonfinite values")
        if arrays["coord"].ndim != 2:
            raise ValueError(f"{sample}: coordinates require a frame axis")
        n = arrays["coord"].shape[0]
        expected = (n, 3 * len(atom_types))
        if n == 0 or arrays["coord"].shape != expected:
            raise ValueError(f"{sample}: unexpected coordinates shape")
        if arrays["force"].shape != expected:
            raise ValueError(f"{sample}: force shape does not match coordinates")
        if arrays["energy"].shape not in ((n,), (n, 1)):
            raise ValueError(f"{sample}: energy count does not match coordinates")
        box_file = sample / "box.npy"
        box = None
        if box_file.is_file():
            box = np.load(box_file, allow_pickle=False)
            if box.shape != (n, 9) or not np.all(np.isfinite(box)):
                raise ValueError(f"{box_file}: invalid cell data")
            determinants = np.linalg.det(box.reshape(n, 3, 3))
            if not np.all(np.abs(determinants) > 1e-6):
                raise ValueError(f"{box_file}: singular periodic cell")
        elif not (path / "nopbc").exists():
            raise ValueError(f"{sample}: missing box.npy for periodic dataset")
        for i in range(n):
            digest = hashlib.sha256()
            digest.update(json.dumps([type_map, atom_types, (path / "nopbc").exists()]).encode())
            digest.update(np.asarray(arrays["coord"][i], dtype="<f8").reshape(-1).tobytes())
            if box is not None:
                digest.update(np.asarray(box[i], dtype="<f8").reshape(-1).tobytes())
            fingerprints.add(digest.hexdigest())
        frames += n
    if frames == 0:
        raise ValueError(f"{path}: no populated set.* directories")
    if dataset_hashes(path) != before_hashes:
        raise ValueError(f"{path}: dataset changed while being inspected; use a stable snapshot")
    return {"path": str(path), "type_map": type_map, "n_atoms": len(atom_types),
            "frames": frames, "geometry_sha256": sorted(fingerprints),
            "files_sha256": before_hashes}


def render_template(template_text: str, train_paths: list[str], steps: int, seed: int, decay: int) -> dict:
    substitutions = {
        "@SEED": str(seed), "@TRAIN_STEPS": str(steps), "@STEPS": str(steps),
        "@DECAY_STEPS": str(decay),
        "@DP_DATASET": ", ".join(json.dumps(x) for x in train_paths),
    }
    for key, value in substitutions.items():
        template_text = template_text.replace(key, value)
    unresolved = sorted(set(re.findall(r"@[A-Z][A-Z0-9_]*", template_text)))
    if unresolved:
        raise ValueError(f"Unresolved template variables: {unresolved}")
    return json.loads(template_text)


def prepare(args: argparse.Namespace) -> dict:
    if args.steps < 1 or args.seed < 0:
        raise ValueError("steps must be positive; seed must be nonnegative")
    if args.decay_steps is not None and args.decay_steps < 1:
        raise ValueError("decay-steps must be positive")
    train = [inspect_system(p) for p in args.train]
    validation = [inspect_system(p) for p in args.validation]
    test = [inspect_system(p) for p in args.test]
    paths = [x["path"] for x in train + validation + test]
    if len(paths) != len(set(paths)):
        raise ValueError("Train/validation/test paths overlap")
    groups = [set().union(*(x["geometry_sha256"] for x in group)) for group in (train, validation, test)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Exact geometry frames overlap across train/validation/test")
    reference_map = train[0]["type_map"]
    if any(x["type_map"] != reference_map for x in train + validation + test):
        raise ValueError("Inconsistent type_map.raw across systems")

    template_path = Path(args.template).expanduser().resolve(strict=True)
    text = template_path.read_text()
    decay = args.decay_steps or max(1, args.steps // 20)
    config = render_template(text, [x["path"] for x in train], args.steps, args.seed, decay)
    if not isinstance(config.get("training"), dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Template needs standard DeePMD model and training sections")
    learning_rate = config.get("learning_rate", {})
    if args.decay_steps is not None:
        if not isinstance(learning_rate, dict) or learning_rate.get("type", "exp") != "exp":
            raise ValueError("--decay-steps is supported only for an exponential learning-rate schedule")
        learning_rate["decay_steps"] = args.decay_steps
        config["learning_rate"] = learning_rate
    # Without an explicit override, keep the literal template schedule. A template
    # placeholder still receives the documented max(1, steps // 20) default.
    config["model"]["type_map"] = reference_map
    descriptor = config["model"].get("descriptor", {})
    sel = descriptor.get("sel") if isinstance(descriptor, dict) else None
    if isinstance(sel, list) and len(sel) != len(reference_map):
        raise ValueError("Descriptor sel length conflicts with element type map")
    training = config["training"]
    training["numb_steps"] = args.steps
    training["seed"] = args.seed
    for component in (descriptor, config["model"].get("fitting_net", {})):
        if isinstance(component, dict) and "seed" in component:
            component["seed"] = args.seed
    training["disp_freq"] = min(training.get("disp_freq", 100), args.steps)
    training["save_freq"] = min(training.get("save_freq", 1000), args.steps)
    training.setdefault("training_data", {})["systems"] = [x["path"] for x in train]
    training["validation_data"] = {
        **training.get("validation_data", {}),
        "systems": [x["path"] for x in validation],
        "batch_size": 1,
    }

    out = Path(args.workdir).expanduser().resolve()
    if (out / "input.json").exists() or (out / "manifest.json").exists():
        raise ValueError(f"{out}: existing input/manifest; use a new run directory")
    out.mkdir(parents=True, exist_ok=True)
    input_text = json.dumps(config, indent=2) + "\n"
    manifest = {
        "schema_version": 2, "input_sha256": hashlib.sha256(input_text.encode()).hexdigest(),
        "effective_learning_rate": config.get("learning_rate", {}),
        "status": "PREPARED_NOT_TRAINED", "template": str(template_path),
        "template_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "type_map": reference_map, "train": train, "validation": validation, "test": test,
        "steps": args.steps, "seed": args.seed,
        "warning": "Check DFT provenance, train/test independence and physical coverage separately.",
    }
    (out / "input.json").write_text(input_text)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--validation", nargs="+", required=True)
    p.add_argument("--test", nargs="+", required=True, help="Independent systems excluded from training")
    p.add_argument("--workdir", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decay-steps", type=int, default=None)
    args = p.parse_args()
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
