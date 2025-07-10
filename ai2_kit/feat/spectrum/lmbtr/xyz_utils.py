# xyz_utils.py

# 1. Single Import Block
import glob
import pickle
import re
import os
from typing import List, Dict, Any, Tuple

import numpy as np
from ase.io import iread
from dscribe.descriptors import LMBTR
from tqdm import tqdm


# Internal helper function (not exposed as a primary utility)
def _get_lmbtr_from_xyzfile(
    filename: str, lmbtrk2: LMBTR, lmbtrk3: LMBTR, central_atoms: List[str]
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Creates LMBTR descriptors for specified central atoms from an XYZ file.
    This is a helper for the main utility function.
    """
    lmbtrs = []
    try:
        for atoms in iread(filename):
            if not central_atoms:
                print("Warning: Central atoms not specified.")
                return []

            central_atom_indices = [
                i for i, symbol in enumerate(atoms.symbols) if symbol in central_atoms
            ]
            for local_index, global_index in enumerate(central_atom_indices):
                lmbtr_descriptor_k2 = lmbtrk2.create(
                    atoms, centers=[global_index], n_jobs=-1
                )
                lmbtr_descriptor_k3 = lmbtrk3.create(
                    atoms, centers=[global_index], n_jobs=-1
                )
                descriptor = np.concatenate(
                    (lmbtr_descriptor_k2[0], lmbtr_descriptor_k3[0]), axis=0
                )
                lmbtrs.append((global_index, local_index, descriptor))
    except Exception as e:
        print(f"Error reading or processing file {filename}: {e}")
    return lmbtrs


# Main utility function
def generate_and_save_lmbtr_descriptors(
    # Data parameters
    xyz_folder_path: str,
    file_pattern: str,
    central_atoms: List[str],
    output_path: str,
    # Control parameters
    save_to_file: bool = False,
    # LMBTR configuration parameters (magic numbers)
    species: List[str] = ["Li", "O", "C", "H", "N", "S", "F"],
    periodic: bool = True,
    k2_grid_min: float = 0,
    k2_grid_max: float = 6,
    k2_grid_n: int = 20,
    k2_grid_sigma: float = 0.1,
    k2_weighting_scale: float = 0.5,
    k2_weighting_threshold: float = 1e-3,
    k3_grid_min: float = 0,
    k3_grid_max: float = 180,
    k3_grid_n: int = 20,
    k3_grid_sigma: float = 0.1,
    k3_weighting_scale: float = 0.5,
    k3_weighting_threshold: float = 1e-3,
    normalization: str = "l2",
):
    """
    Reads XYZ files, generates LMBTR descriptors for specified atoms,
    and optionally saves the collected descriptors to a pickle file.

    Args:
        xyz_folder_path (str): Path to the directory containing XYZ files.
                               Example: './data/xyz/'
        file_pattern (str): Glob pattern to match files within the folder.
                            Example: 'task-*.xyz'
        central_atoms (List[str]): List of central atom symbols (e.g., ['Li']).
        output_path (str): Full path for the output pickle file, including filename.
                           Example: './descriptors/fsi.pkl'
        save_to_file (bool): If True, saves the descriptors to a pickle file.
                             Defaults to False.
        species (List[str]): List of all species present in the systems.
        periodic (bool): Whether the system is periodic.
        k2_grid_min (float): Minimum for the k=2 grid (distance).
        k2_grid_max (float): Maximum for the k=2 grid (distance).
        k2_grid_n (int): Number of points for the k=2 grid.
        k2_grid_sigma (float): Smearing for the k=2 grid.
        k2_weighting_scale (float): Scale for the k=2 exponential weighting.
        k2_weighting_threshold (float): Cutoff threshold for k=2 weighting.
        k3_grid_min (float): Minimum for the k=3 grid (angle).
        k3_grid_max (float): Maximum for the k=3 grid (angle).
        k3_grid_n (int): Number of points for the k=3 grid.
        k3_grid_sigma (float): Smearing for the k=3 grid.
        k3_weighting_scale (float): Scale for the k=3 exponential weighting.
        k3_weighting_threshold (float): Cutoff threshold for k=3 weighting.
        normalization (str): Normalization type for LMBTR.

    Returns:
        List[np.ndarray]: A list containing all the generated descriptor arrays.
    """
    # Define LMBTR descriptors based on parameterized inputs
    lmbtrk2 = LMBTR(
        species=species,
        geometry={"function": "distance"},
        grid={
            "min": k2_grid_min,
            "max": k2_grid_max,
            "n": k2_grid_n,
            "sigma": k2_grid_sigma,
        },
        weighting={
            "function": "exp",
            "scale": k2_weighting_scale,
            "threshold": k2_weighting_threshold,
        },
        periodic=periodic,
        normalization=normalization,
    )
    lmbtrk3 = LMBTR(
        species=species,
        geometry={"function": "angle"},
        grid={
            "min": k3_grid_min,
            "max": k3_grid_max,
            "n": k3_grid_n,
            "sigma": k3_grid_sigma,
        },
        weighting={
            "function": "exp",
            "scale": k3_weighting_scale,
            "threshold": k3_weighting_threshold,
        },
        periodic=periodic,
        normalization=normalization,
    )

    # Process files
    search_path = os.path.join(xyz_folder_path, file_pattern)
    filename_ensemble = sorted(glob.glob(search_path))
    all_descriptors = []

    print(f"Found {len(filename_ensemble)} file(s) to process.")

    for filename in tqdm(filename_ensemble):
        descriptors_with_indices = _get_lmbtr_from_xyzfile(
            filename, lmbtrk2, lmbtrk3, central_atoms
        )
        # Extract only the descriptor array, as in the original script
        for _, _, descriptor in descriptors_with_indices:
            all_descriptors.append(descriptor)

    # Handle side effect: saving to file
    if save_to_file:
        file_index_des = {
            "descriptor": all_descriptors,
        }
        with open(output_path, "wb") as f:
            pickle.dump(file_index_des, f)
        print(f"Descriptors saved to {output_path}")

    return all_descriptors