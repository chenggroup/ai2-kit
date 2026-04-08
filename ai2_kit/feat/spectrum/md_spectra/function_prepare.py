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

def get_distance(coords_A: np.ndarray, coords_B: Optional[np.ndarray], cells: np.ndarray, 
                 remove_diag: bool = False, offset: int = 0):
    """
        Calculate the distances between coords_A and coords_B.
        The distance is calculated in the sense of PBC.

        Parameters
        -------------
            coords_A (..., num_A, 3): the coordinates of the central points. The size of the last axis is the dimension.
            coords_B (..., num_B, 3): the coordinates of the points to be selected. 
            If B is None, A will be compared with itself.
            cells    (..., 3, 3): the PBC cells.
            remove_diag, bool: whether to fill the diagonal with np.inf.

        Return
        -------------
            distance (..., num_A, num_B): the matrix of distances.
    """
    if coords_B is None:
        coords_B = coords_A
    distance = np.linalg.norm(
        box_shift(
            coords_A[..., None, :] - coords_B[..., None, :, :],  # type: ignore
            cells
        ), 
        ord = 2, axis = -1
    )
    if remove_diag:
        write_to_diagonal(distance, np.inf, offset = offset, axis1 = -2, axis2 = -1)
    return distance

def _coords_cells_mul(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    if coords.ndim >= cells.ndim:
        d0 = coords.ndim - cells.ndim + 1
        _shape = coords.shape
        return np.matmul(coords.reshape(_shape[:-d0-1] + (-1, 3)), cells).reshape(_shape)
    else:
        return np.matmul(coords[..., None, :], cells).squeeze(-2)
def inv_cells(cells: np.ndarray):
    """
    Reciprocal cells.
    """
    return np.linalg.inv(cells)
def to_frac(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """
    Transfer from the cartesian coordinate to fractional coordinate.

    Parameters
    -----
    coords: np.ndarray,
    in shape of (..., 3)

    cells: np.ndarray,
    in shape of (..., 3, 3)

    Return
    -----
    fractional coords: np.ndarray,
    in shape of (..., 3)
    """
    recip_cell = inv_cells(cells)
    return _coords_cells_mul(coords, recip_cell)

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
    return dx - _coords_cells_mul(np.round(to_frac(dx, cells)), cells)

def do_pbc(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    '''
    Translate to the home cell.

    Parameters
    -----
    coords: np.ndarray,
    in shape of (..., natom, 3)

    cells: np.ndarray,
    in shape of (..., 3, 3)

    Return
    -----
    translated coords: np.ndarray,
    in shape of (..., 3)
    '''
    return coords - _coords_cells_mul(np.floor(to_frac(coords, cells)), cells)

def k_nearest(coords_A: np.ndarray, coords_B: Optional[np.ndarray], cells: np.ndarray, 
                   k: int, batch_size: int = -1):
    """
    For each atom in coords_A, choose the k-nearest atoms (in the cell) among coords_B, and return their indices.
    The distance is calculated in the sense of PBC.

    Parameters
    -------------
    coords_A (..., num_A, 3): 
    the coordinates of the central points. The size of the last axis is the dimension.

    coords_B (..., num_B, 3): 
    the coordinates of the points to be selected. 
    If B is None, A will be compared with itself, where the diagonal will be removed.

    cells (..., 3, 3): 
    the PBC box. cells[..., :, i] is the i-th axis of the cell.

    k: int, the number of the points selected from coords_B.

    batch_size: int, the batch size of atoms in coords_B at each time. 
    The required memory size is (..., num_A, k + batch_size).
    If batch_size <= 0, it will use the largest batch_size, 
    which means the required memory size is (..., num_A, num_B).

    Return
    -------------
    indices (..., num_A, k): the indices of the k-nearest points in coords_B.

    distances (..., num_A, k): the distances of the k-nearest points in coords_B.
    """
    self_comp = False
    if coords_B is None:
        coords_B = coords_A
        self_comp = True
    d = coords_B.shape[-2]
    k = min(d, k)
    batch_size = min(d - k, batch_size)
    if batch_size <= 0:
        distance = get_distance(coords_A, coords_B, cells, remove_diag = self_comp)
        k_index = np.argpartition(distance, k, axis = -1)[..., :k]
        k_distance = np.take_along_axis(distance, k_index, axis = -1)
    else:
        _shape = list(coords_A.shape)
        _shape[-1] = k + batch_size
        k_index = np.empty(_shape, dtype = int)
        k_distance = np.empty(_shape, dtype = coords_B.dtype)
        k_index[..., :k] = np.arange(k)
        k_distance[..., :k] = get_distance(
            coords_A, coords_B[..., :k, :], cells, remove_diag = self_comp, offset = 0
        )
        for i in range(k, d, batch_size):
            end_i = min(d, i + batch_size)
            sz = end_i - i
            k_index[..., k:k + sz] = np.arange(i, end_i)
            k_distance[..., k:k + sz] = get_distance(
                coords_A, coords_B[..., i:end_i, :], cells, remove_diag = self_comp, offset = i
            )
            partition_idx = np.argpartition(k_distance, k, axis = -1)
            k_index = np.take_along_axis(k_index, partition_idx, axis = -1)
            k_distance = np.take_along_axis(k_distance, partition_idx, axis = -1)
    sort_idx = np.argsort(k_distance[..., :k], axis = -1)
    k_index = np.take_along_axis(k_index[..., :k], sort_idx, axis = -1)
    k_distance = np.take_along_axis(k_distance[..., :k], sort_idx, axis = -1)
    return k_index

def find_h2o(coords_sel: np.ndarray, coords_oth: np.ndarray, cells: np.ndarray, 
                r_bond: float, 
                # mask_sel: Optional[np.ndarray] = None
                ):
    coords_sel = coords_sel[..., np.newaxis, :]
    # (..., num_sel, num_oth, 3)
    delta = box_shift(coords_oth[..., np.newaxis, :, :] - coords_sel, cells[..., np.newaxis, np.newaxis, :, :]) # type: ignore
    mask = np.linalg.norm(delta, 2, axis = -1) < r_bond
    h2o_mask = np.sum(mask, axis = -1) == 2
    return h2o_mask

def calculate_dipole_H(coords_O: np.ndarray, coords_H: np.ndarray, cells: np.ndarray):
    idx_H = k_nearest(coords_O[[0]], coords_H[[0]], cells[[0]], 2)
    cH = np.take_along_axis(coords_H[..., np.newaxis, :, :], idx_H[..., np.newaxis], axis = -2)
    return (np.sum(box_shift(cH - coords_O[..., np.newaxis, :], cells[..., np.newaxis, np.newaxis, :, :]), axis = -2))

def calculate_dipole(coords_O: np.ndarray, coords_H: np.ndarray, cells: np.ndarray, wannier: np.ndarray) -> np.ndarray:
    dipole_H = calculate_dipole_H(coords_O, coords_H, cells)
    atomic_dipole = dipole_H - wannier * 8
    return atomic_dipole