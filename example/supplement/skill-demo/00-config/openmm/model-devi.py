#!/usr/bin/env python
"""
Compute multi-model force deviation for MACE models on a trajectory.

Outputs a DeePMD-kit-compatible model_devi.out file for use with
`ai2-kit tool model_devi` during the TESLA screening phase.

Usage:
    python model-devi.py <traj_file> <output_file> <model1.model> [model2.model ...]

Arguments:
    traj_file     Trajectory file to analyse (extxyz format)
    output_file   Path for the resulting model_devi.out
    model*.model  One or more MACE model files; at least two are required for a
                  meaningful deviation estimate.

Output columns (DeePMD-kit convention):
    step  max_devi_v  min_devi_v  avg_devi_v  max_devi_f  min_devi_f  avg_devi_f

    Virial deviation (max/min/avg_devi_v) is set to a constant placeholder
    (1.0) because screening relies exclusively on force deviation.
    Force deviation per atom is defined as the L2 norm of the per-component
    standard deviation across all models:
        devi_f_i = || std_{models}( F_i ) ||_2
    max/min/avg_devi_f are then the max/min/mean over all atoms.
"""

import sys
import numpy as np
import ase.io
from mace.calculators import MACECalculator

VIRIAL_PLACEHOLDER = 1.0   # not used in screening; set to a high constant


def compute_model_devi(traj_file: str, output_file: str, model_paths: list[str]):
    # -----------------------------------------------------------------------
    # Load trajectory
    # -----------------------------------------------------------------------
    frames = ase.io.read(traj_file, index=':', format='extxyz')
    print(f"Loaded {len(frames)} frames from '{traj_file}'")
    print(f"Computing deviations across {len(model_paths)} models...")

    # Pre-load one calculator per model (avoids repeated model initialisation)
    calcs = [
        MACECalculator(
            model_paths=[m],
            device='cuda',
            default_dtype='float64',
        )
        for m in model_paths
    ]

    results = []
    for idx, atoms in enumerate(frames):
        all_forces = []
        for calc in calcs:
            tmp = atoms.copy()
            tmp.calc = calc
            all_forces.append(tmp.get_forces())  # eV/Å, shape (n_atoms, 3)

        all_forces = np.array(all_forces)  # (n_models, n_atoms, 3)

        # Per-component std across models, then L2 norm → per-atom deviation
        force_std      = np.std(all_forces, axis=0)          # (n_atoms, 3)
        force_devi_per_atom = np.linalg.norm(force_std, axis=-1)  # (n_atoms,)

        results.append((
            idx,
            VIRIAL_PLACEHOLDER, VIRIAL_PLACEHOLDER, VIRIAL_PLACEHOLDER,
            float(force_devi_per_atom.max()),
            float(force_devi_per_atom.min()),
            float(force_devi_per_atom.mean()),
        ))

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(frames)} frames")

    # -----------------------------------------------------------------------
    # Write model_devi.out
    # -----------------------------------------------------------------------
    header = (
        '#       step         max_devi_v         min_devi_v'
        '         avg_devi_v         max_devi_f         min_devi_f         avg_devi_f'
    )
    with open(output_file, 'w') as f:
        f.write(header + '\n')
        for row in results:
            f.write(
                f"{row[0]:>12d}"
                f" {row[1]:>18.6e} {row[2]:>18.6e} {row[3]:>18.6e}"
                f" {row[4]:>18.6e} {row[5]:>18.6e} {row[6]:>18.6e}\n"
            )
    print(f"Model deviation written to '{output_file}'")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    traj_file   = sys.argv[1]
    output_file = sys.argv[2]
    model_paths = sys.argv[3:]

    compute_model_devi(traj_file, output_file, model_paths)
