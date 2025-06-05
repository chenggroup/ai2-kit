from typing import Optional
import numpy as np

def write_to_diagonal(a: np.ndarray, diag: np.ndarray, offset: int = 0, axis1: int = 0, axis2: int = 1):
    diag_slices = [slice(None) for _ in a.shape]
    start_idx = [max(-offset, 0), max(offset, 0)]
    diag_len = min(a.shape[axis1] - start_idx[0], a.shape[axis2] - start_idx[1])
    assert diag_len >= 0
    if diag_len == 0:
        return
    diag_slices[axis1] = list(range(start_idx[0], start_idx[0] + diag_len))
    diag_slices[axis2] = list(range(start_idx[1], start_idx[1] + diag_len))
    a[tuple(diag_slices)] = diag

def to_frac(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    recip_cell = np.zeros_like(cells)
    recip_cell[..., :, 0] = np.cross(cells[..., 1, :], cells[..., 2, :])
    recip_cell[..., :, 1] = np.cross(cells[..., 2, :], cells[..., 0, :])
    recip_cell[..., :, 2] = np.cross(cells[..., 0, :], cells[..., 1, :])
    vol = np.sum(recip_cell[..., :, 0] * cells[..., 0, :], axis = -1)
    recip_cell /= vol[..., np.newaxis, np.newaxis]
    return np.sum(coords[..., np.newaxis] * recip_cell, axis = -2)

def box_shift(dx: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """
    Shift the coordinates (dx) to the coordinates that have the smallest absolute value.

    Parameters
    -----
    dx: np.ndarray,
    in shape of (..., 3)

    cells: np.ndarray,
    in shape of (..., 3, 3)

    Return
    -----
    shifted_dx: np.ndarray,
    in shape of (..., 3)
    """
    frac_c = to_frac(dx, cells)[..., np.newaxis]            # (..., 3, 1)
    return dx - np.sum(np.round(frac_c) * cells, axis = -2) # (..., 3)

def k_nearest(coords_A: np.ndarray, coords_B: Optional[np.ndarray], cells: np.ndarray, k: int):
    """
        For each point in coords_A, choose the k-nearest points (in the box) among coords_B, and return the index.
        The distance is calculated in the sense of PBC.

        Parameters
        -------------
            coords_A (..., num_A, d): the coordinates of the central points. The size of the last axis is the dimension.
            coords_B (..., num_B, d): the coordinates of the points to be selected.
            box      (..., d): the PBC box. box[..., i] is the length of period along x_i.
            k: int, the number of the points selected from coords_B.

        Return
        -------------
            index (..., num_A, k): the index of the k-nearest points in coords_B.
    """
    self_comp = False
    if coords_B is None:
        coords_B = coords_A
        self_comp = True
    distance = np.linalg.norm(
        box_shift(
            coords_A[..., np.newaxis, :] - coords_B[..., np.newaxis, :, :], 
            cells[..., np.newaxis, np.newaxis, :, :]
        ), 
        ord = 2, axis = -1
    )
    if self_comp:
        write_to_diagonal(distance, np.inf, offset = 0, axis1 = -2, axis2 = -1)
    return np.argsort(distance, axis = -1)[..., :k]

def do_pbc(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    '''
    Translate to the home cell.

    Parameters
    -----
    coords: np.ndarray,
    in shape of (..., natom, 3)

    cells: np.ndarray,
    in shape of (..., 3    , 3)

    Return
    -----
    translated coords: np.ndarray,
    in shape of (..., 3)
    '''
    _cells = cells[..., np.newaxis, :, :]       # TODO
    frac_c = to_frac(coords, _cells)[..., np.newaxis]
    return coords - np.sum(np.floor(frac_c) * _cells, axis = -2)

def find_h2o(coords_sel: np.ndarray, coords_oth: np.ndarray, cells: np.ndarray, 
                r_bond: float, 
                # mask_sel: Optional[np.ndarray] = None
                ):
    coords_sel = coords_sel[..., np.newaxis, :]
    # (..., num_sel, num_oth, 3)
    delta = box_shift(coords_oth[..., np.newaxis, :, :] - coords_sel, cells[..., np.newaxis, np.newaxis, :, :]) # type: ignore
    mask = np.linalg.norm(delta, 2, axis = -1) < r_bond
    h2o_mask = np.sum(mask, axis = -1) >= 2
    return h2o_mask

def calculate_dipole_H(coords_O: np.ndarray, coords_H: np.ndarray, cells: np.ndarray):
    idx = k_nearest(coords_O[[0]], coords_H[[0]], cells[[0]], 2)
    cH = np.take_along_axis(coords_H[..., np.newaxis, :, :], idx[..., np.newaxis], axis = -2)
    return np.sum(box_shift(cH - coords_O[..., np.newaxis, :], cells[..., np.newaxis, np.newaxis, :, :]), axis = -2)

def calculate_dipole(coords_O: np.ndarray, coords_H: np.ndarray, cells: np.ndarray, wannier: np.ndarray, r_bond = 1.1) -> np.ndarray:
    dipole_H = calculate_dipole_H(coords_O, coords_H, cells)
    atomic_dipole = dipole_H - wannier * 8
    # total_dipole = np.sum(atomic_dipole, axis = 1)
    vol = np.linalg.det(cells)[..., np.newaxis, np.newaxis]
    return atomic_dipole / np.sqrt(vol / coords_O.shape[1])