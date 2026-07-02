#!/usr/bin/env python3
"""
Calculate model deviation for forces using multiple MACE models.
Outputs in DeePMD-compatible format (force columns only; virial columns
are set to a large placeholder so that ai2-kit screening ignores them).

Usage:
    python model-devi.py model1.model model2.model model3.model \
        [--traj traj.xyz] [--output model_devi.out] \
        [--device cpu|cuda] [--fmt <ase-format>]
"""

import argparse
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

# Placeholder value written to virial columns (not computed)
_LARGE = 9.999999e+08


def _devi_f(forces_list):
    """
    Compute per-frame force model deviation statistics.

    Parameters
    ----------
    forces_list : list of np.ndarray, shape (N_atoms, 3)
        Force arrays predicted by each model for a single frame.

    Returns
    -------
    max_devi_f, min_devi_f, avg_devi_f : float
        Maximum, minimum, and mean per-atom force deviation (eV/Å).
    """
    # (n_models, N_atoms, 3)
    forces = np.stack(forces_list, axis=0)
    # Mean force across models: (N_atoms, 3)
    mean_f = forces.mean(axis=0)
    # Per-model, per-atom deviation norm: (n_models, N_atoms)
    deviations = np.linalg.norm(forces - mean_f[np.newaxis], axis=-1)
    # Per-atom deviation = max over models
    per_atom = deviations.max(axis=0)  # (N_atoms,)
    return per_atom.max(), per_atom.min(), per_atom.mean()


def main():
    parser = argparse.ArgumentParser(
        description="Compute MACE model deviation and write DeePMD-compatible output."
    )
    parser.add_argument("models", nargs="+", help="Paths to MACE model files (.pt)")
    parser.add_argument("--traj",   default="traj.xyz",       help="Trajectory file (default: traj.xyz)")
    parser.add_argument("--output", default="model_devi.out", help="Output file (default: model_devi.out)")
    parser.add_argument("--device", default="cpu",            help="Compute device: cpu or cuda (default: cpu)")
    parser.add_argument("--fmt",    default=None,             help="ASE trajectory format (auto-detected if omitted)")
    args = parser.parse_args()

    if len(args.models) < 2:
        raise ValueError("At least 2 models are required to compute model deviation.")

    # ------------------------------------------------------------------
    # Load trajectory
    # ------------------------------------------------------------------
    print(f"Reading trajectory: {args.traj}")
    frames = read(args.traj, index=":", format=args.fmt)
    print(f"  {len(frames)} frames loaded.")

    # ------------------------------------------------------------------
    # Build one MACECalculator per model
    # MACECalculator accepts a single path string for model_paths.
    # ------------------------------------------------------------------
    print(f"Loading {len(args.models)} MACE models on device={args.device} ...")
    calculators = [
        MACECalculator(model_paths=m, device=args.device, default_dtype="float64")
        for m in args.models
    ]

    # ------------------------------------------------------------------
    # Evaluate and write output
    # ------------------------------------------------------------------
    header = (
        f"{'#':>10s}{'step':>18s}"
        f"{'max_devi_v':>18s}{'min_devi_v':>18s}{'avg_devi_v':>18s}"
        f"{'max_devi_f':>18s}{'min_devi_f':>18s}{'avg_devi_f':>18s}\n"
    )

    print(f"Writing model deviation to: {args.output}")
    with open(args.output, "w") as fout:
        fout.write(header)
        for step, atoms in enumerate(frames):
            forces_list = []
            for calc in calculators:
                atoms.calc = calc
                forces_list.append(atoms.get_forces().copy())

            max_f, min_f, avg_f = _devi_f(forces_list)

            fout.write(
                f"{'':>10s}{step:>18d}"
                f"{_LARGE:>18.6e}{_LARGE:>18.6e}{_LARGE:>18.6e}"
                f"{max_f:>18.6e}{min_f:>18.6e}{avg_f:>18.6e}\n"
            )

            if (step + 1) % 100 == 0:
                print(f"  {step + 1}/{len(frames)} frames done ...")

    print("Done.")


if __name__ == "__main__":
    main()
