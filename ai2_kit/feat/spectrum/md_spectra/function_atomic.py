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

def find_oh(coords_sel: np.ndarray, coords_oth: np.ndarray, cells: np.ndarray, 
                r_bond: float, 
                # mask_sel: Optional[np.ndarray] = None
                ):
    coords_sel = coords_sel[..., np.newaxis, :]
    # (..., num_sel, num_oth, 3)
    delta = box_shift(coords_oth[..., np.newaxis, :, :] - coords_sel, cells[..., np.newaxis, np.newaxis, :, :]) # type: ignore
    mask = np.linalg.norm(delta, 2, axis = -1) < r_bond
    oh_mask = np.sum(mask, axis = -1) == 1
    return oh_mask

def calculate_dipole_H(coords_O: np.ndarray, coords_H: np.ndarray, cells: np.ndarray):
    idx_H = k_nearest(coords_O[[0]], coords_H[[0]], cells[[0]], 2)
    cH = np.take_along_axis(coords_H[..., np.newaxis, :, :], idx_H[..., np.newaxis], axis = -2)
    return (np.sum(box_shift(cH - coords_O[..., np.newaxis, :], cells[..., np.newaxis, np.newaxis, :, :]), axis = -2))

def calculate_dipole(coords_O: np.ndarray, coords_H: np.ndarray, cells: np.ndarray, wannier: np.ndarray) -> np.ndarray:
    dipole_H = calculate_dipole_H(coords_O, coords_H, cells)
    atomic_dipole = dipole_H - wannier * 8
    return atomic_dipole

def calculate_dipole_OH_H(coords_O: np.ndarray, coords_H: np.ndarray, coords_Al: np.ndarray, cells: np.ndarray):
    idx_H = k_nearest(coords_O[[0]], coords_H[[0]], cells[[0]], 1)
    idx_Al = k_nearest(coords_O[[0]], coords_Al[[0]], cells[[0]], 2)
    cH = np.take_along_axis(coords_H[..., np.newaxis, :, :], idx_H[..., np.newaxis], axis = -2)
    cAl = np.take_along_axis(coords_Al[..., np.newaxis, :, :], idx_Al[..., np.newaxis], axis = -2)
    return (np.sum(box_shift(cH - coords_O[..., np.newaxis, :], cells[..., np.newaxis, np.newaxis, :, :]), axis = -2) 
            + np.sum(box_shift(cAl - coords_O[..., np.newaxis, :], cells[..., np.newaxis, np.newaxis, :, :]), axis = -2) / 6)

def calculate_dipole_OH(coords_O: np.ndarray, coords_H: np.ndarray, coords_Al: np.ndarray, cells: np.ndarray, wannier: np.ndarray) -> np.ndarray:
    dipole_H = calculate_dipole_OH_H(coords_O, coords_H, coords_Al, cells)
    atomic_dipole = dipole_H - wannier * 8
    return atomic_dipole

def calculate_corr(A: np.ndarray, B: np.ndarray, NMAX: int, window: Optional[int] = None):
    """
    Calculate the correlation function: `corr(t) = <A(0) * B(t)>`. 
    Here, `A(t)` and `B(t)` are arrays of the same dimensions, 
    and `A(t) * B(t)` is the element-wise multiplication. 
    The esenmble average `< >` is estimated by moving average.

    Parameters
    -----
    A, B: np.ndarray, in shape of (num_t, ...).
        The first dimension refers to the time steps, and its size can be different.
        The remaining dimensions (if present) must be the same.
    
    NMAX: int.
        Maximal time steps. Calculate `corr(t)` with `0 <= t <= NMAX`.
    
    window: int, optional.
        The width of window to do the moving average. 

        `<A(0) * B(t)> = 1 / window * \sum_{i = 0}^{window - 1} A(i) * B(t + i)`. 

    Return
    -----
    corr: np.ndarray, in shape of (NMAX + 1, ...).
        `corr(t) = <A(0) * B(t)>`
    """
    if A.ndim == 1 or B.ndim == 1:
        A = A.reshape(-1, 1)
        B = B.reshape(-1, 1)
    if window is None:
        window = min(A.shape[0], B.shape[0] - NMAX)
    # Prepare for convolution
    v1 = A[:window][::-1]; v2 = B[:window + NMAX]
    pad_width = [(0, 0)] * A.ndim
    pad_width[0] = (0, NMAX)
    v1 = np.pad(v1, pad_width, "constant", constant_values = 0)
    # Convolve by FFT
    corr = np.fft.ifft(np.fft.fft(v1, axis = 0) * np.fft.fft(v2, axis = 0), axis = 0).real # type: ignore
    # Moving average
    corr = corr[window - 1:window + NMAX] / window
    return corr

def cutoff_ir_raman(arr, low, high, smooth_width):
    eps = 1e-2
    cut_f = lambda x: np.exp(-1/np.clip(x, eps, None))
    a_in = cut_f(np.maximum(low - arr, arr - high))
    a_out = cut_f(np.minimum(arr - low, high - arr) + smooth_width)
    return a_out / (a_out + a_in)

def cal_range_dipole_polar(z, atomic_dipole, z_lo, z_hi, r_smth):
    weight = cutoff_ir_raman(z, z_lo, z_hi, r_smth)
    range_dipole = np.sum(weight * atomic_dipole, axis = 1) / np.sqrt(np.clip(np.sum(weight, axis = 1), 1, None))
    return range_dipole

def calculate_corr_vdipole_atomic(atomic_dipole: np.ndarray, weight: np.ndarray, coords: np.ndarray, 
                           cells: np.ndarray, dt_ps: float, window: int, rc: float = 6.75):
    nframes, natom = atomic_dipole.shape[:2]
    weight = weight[1:-1]
    coords = coords[1:-1]
    cells = cells[1:-1]
    weight /= np.sqrt(np.clip(np.sum(np.abs(weight), axis = 1, keepdims = True), 1, None))
    v_dipole = weight[..., None] * (atomic_dipole[2:] - atomic_dipole[:-2]) / (2 * dt_ps)
    # v_dipole -= np.mean(v_dipole, axis = 0, keepdims = True)
    corr_intra = calculate_corr(v_dipole, v_dipole, window)
    dipole_cutoff = np.empty_like(v_dipole)
    for atom_i in range(natom):
        dis_mask = get_distance(coords, coords[:, [atom_i], :], cells) < rc
        dis_mask[:, atom_i] = False
        dipole_cutoff[:, atom_i] = np.matmul(v_dipole.transpose(0, 2, 1), dis_mask).squeeze(2)
    corr_inter = calculate_corr(dipole_cutoff, v_dipole, window)
    return corr_intra, corr_inter

def calculate_corr_polar_atomic(atomic_polar: np.ndarray, weight: np.ndarray, coords: np.ndarray, 
                         cells: np.ndarray, window: int, rc: float = 6.75):
    nframes, natom = atomic_polar.shape[:2]

    polar_iso = np.mean(atomic_polar.diagonal(offset = 0, axis1 = -2, axis2 = -1), axis = -1)
    diag = np.zeros_like(atomic_polar, dtype = float)
    diag[..., 0, 0] = polar_iso
    diag[..., 1, 1] = polar_iso
    diag[..., 2, 2] = polar_iso
    polar_aniso = atomic_polar - diag # type: ignore

    polar_iso -= np.mean(polar_iso, axis = 0, keepdims = True)
    polar_aniso -= np.mean(polar_aniso, axis = 0, keepdims = True)
    polar_iso = np.square(weight) * polar_iso
    polar_aniso = np.square(weight[..., None, None]) * polar_aniso
    polar_aniso = polar_aniso.reshape(nframes, natom, 9)

    corr_iso_intra = calculate_corr(polar_iso, polar_iso, window)
    corr_aniso_intra = np.sum(calculate_corr(polar_aniso, polar_aniso, window), axis = -1) * (2. / 15.)
    
    polar_iso_cutoff = np.empty_like(polar_iso)
    polar_aniso_cutoff = np.empty_like(polar_aniso)
    for atom_i in range(natom):
        dis_mask = get_distance(coords, coords[:, [atom_i], :], cells) < rc
        dis_mask[:, atom_i] = False
        polar_iso_cutoff[:, atom_i] = np.matmul(polar_iso[:, None, :], dis_mask).squeeze((1, 2))
        polar_aniso_cutoff[:, atom_i] = np.matmul(polar_aniso.transpose(0, 2, 1), dis_mask).squeeze(2)
    corr_iso_inter = calculate_corr(polar_iso_cutoff, polar_iso, window)
    corr_aniso_inter = np.sum(calculate_corr(polar_aniso_cutoff, polar_aniso, window), axis = -1) * (2. / 15.)

    return corr_iso_intra, corr_aniso_intra, corr_iso_inter, corr_aniso_inter

def cutoff_z(z_arr, z0, zc, zw):
    """Cutoff along z-axis by cosine function."""
    cut_f = lambda x: np.cos(np.pi * np.clip(x, -1, 0) / 2)**2
    z = z_arr - z0
    return np.sign(z) * cut_f((np.abs(z) - (zc + zw)) / zw)

# def zaxis_weight(z_arr, z0, zc, zw):
#     weight = cutoff_z(z_arr, z0, zc, zw)
#     return weight / np.sqrt(np.clip(np.sum(np.abs(weight), axis = 1, keepdims = True), 1, None))

# def cal_weighted_dipole(weight, atomic_dipole):
#     return weight * atomic_dipole

# def cal_weighted_polar(weight, atomic_polar):
#     return (weight ** 2) * atomic_polar

def cal_corr_sfg_method1(atomic_polar: np.ndarray, atomic_dipole: np.ndarray, weight: np.ndarray, 
                         coords: np.ndarray, cells: np.ndarray, window: int, rc: float = 6.75):
    """
    Calculate SFG correlation by `S(0)\mu(0)S(t)^2\alpha(t)`.
    """
    nframes, natom = atomic_dipole.shape[:2]
    atomic_polar -= np.mean(atomic_polar, axis = 0, keepdims = True)
    atomic_dipole -= np.mean(atomic_dipole, axis = 0, keepdims = True)
    dipole = weight * atomic_dipole    
    polar = np.square(weight) * atomic_polar
    corr_intra = calculate_corr(dipole, polar, window)
    dipole_cutoff = np.empty_like(dipole)
    for atom_i in range(natom):
        dis_mask = get_distance(coords, coords[:, [atom_i], :], cells) < rc
        dis_mask[:, atom_i] = False
        dipole_cutoff[:, atom_i] = np.matmul(dipole[:, None, :], dis_mask).squeeze((1, 2))
    corr_inter = calculate_corr(dipole_cutoff, polar, window)
    return corr_intra, corr_inter

def cal_corr_sfg_method2(atomic_polar: np.ndarray, atomic_dipole: np.ndarray, weight: np.ndarray, 
                         coords: np.ndarray, cells: np.ndarray, window: int, rc: float = 6.75):
    """
    Calculate SFG correlation by `S(0)\mu(0)S(0)^2\alpha(t)`.
    """
    nframes, natom = atomic_dipole.shape[:2]
    atomic_polar -= np.mean(atomic_polar, axis = 0, keepdims = True)
    atomic_dipole -= np.mean(atomic_dipole, axis = 0, keepdims = True)
    dipole = weight * atomic_dipole
    polar = atomic_polar
    corr_intra = calculate_corr(dipole, np.square(weight) * polar, window)
    dipole_cutoff = np.empty_like(dipole)
    for atom_i in range(natom):
        dis_mask = get_distance(coords, coords[:, [atom_i], :], cells) < rc
        dis_mask[:, atom_i] = False
        dipole_cutoff[:, atom_i] = np.matmul(dipole[:, None, :], dis_mask).squeeze() * np.square(weight[:, atom_i])
    corr_inter = calculate_corr(dipole_cutoff, polar, window)
    return corr_intra, corr_inter

def apply_gussian_filter(corr: np.ndarray, width: float):
    """
    Apply gaussian filter. Parameter `width` means the smoothing width.
    """
    nmax = corr.shape[0] - 1
    return corr * np.exp(-.5 * (0.5 * width * np.arange(nmax + 1) / nmax)**2)

def apply_lorenz_filter(corr: np.ndarray, width: float, dt):
    """
    Apply Cauchy-Lorenz filter. Parameter `width` means the smoothing width.
    """
    nmax = corr.shape[0] - 1
    b = width * 2.99792458e-3
    M = int(1 / (dt * 0.01 * 2)) * 2
    M = max(M, nmax)
    dx = 1 / (M * dt)
    NX = int(50 * np.sqrt(b) / dx / 2) * 2
    x = np.arange(NX + 1) * dx
    p = b / (b**2 + x**2)
    _, ph = FT(dx, p, M)
    return corr * ph[:nmax + 1]

def FT(DT: float, C: np.ndarray, M: Optional[int] = None) -> np.ndarray:
    """
    Perform a cosine transform on the correlation function using FFT.
    The same as FILONC while `DOM = 2\pi / (M * DT)` (or `OMEGA_MAX = 2\pi / DT`).

    Parameters
    -----
    C: ndarray
        the correlation function.
    DT: float
        time interval between points in C.
    M: Optional[int]
        number of intervals on the frequency axis. Default is `len(corr) - 1`.

    Returns
    -----
    Frequency and the 1-d cosine transform of the correlation function.
    freq: float, frequency. `freq = 1 / (M * DT)` 
    CHAT: np.ndarray, the 1-d cosine transform.
    """
    NMAX = C.shape[0] - 1
    if NMAX % 2 != 0:
        raise ValueError("NMAX (=len(C)-1) must be even for the cosine transform.")
    if M is None:
        M = NMAX
    elif M % 2 != 0:
        M += 1
        
    freq = 1 / (M * DT)
    DTH = 2 * np.pi / M
    NU = np.arange(M + 1)
    THETA = NU * DTH

    ALPHA, BETA, GAMMA = _FILON_PARAMS(THETA)
    CE, CO = _FFT_OE(C, M)
    CE, CO = CE.real, CO.real
    CE -= 0.5 * (C[0] + C[NMAX] * np.cos(THETA * NMAX))

    CHAT = 2.0 * (ALPHA * C[NMAX] * np.sin (THETA * NMAX) + BETA * CE + GAMMA * CO) * DT
    return freq, CHAT

def FT_sin(DT: float, C: np.ndarray, M: Optional[int] = None) -> np.ndarray:
    """
    Perform a sine transform on the correlation function using FFT.

    Parameters
    -----
    C: ndarray
        the correlation function.
    DT: float
        time interval between points in C.
    M: Optional[int]
        number of intervals on the frequency axis. Default is `len(corr) - 1`.

    Returns
    -----
    Frequency and the 1-d sine transform of the correlation function.
    freq: float, frequency. `freq = 1 / (M * DT)` 
    CHAT: np.ndarray, the 1-d sine transform.
    """
    NMAX = C.shape[0] - 1
    if NMAX % 2 != 0:
        raise ValueError("NMAX (=len(C)-1) must be even for the sine transform.")
    if M is None:
        M = NMAX
    elif M % 2 != 0:
        M += 1
    
    freq = 1 / (M * DT)
    DTH = 2 * np.pi / M
    NU = np.arange(M + 1)
    THETA = NU * DTH

    ALPHA, BETA, GAMMA = _FILON_PARAMS(THETA)
    CE, CO = _FFT_OE(C, M)
    CE, CO = CE.imag, CO.imag
    CE -= 0.5 * (C[NMAX] * np.sin(THETA * NMAX))

    CHAT = 2.0 * (ALPHA * (C[0] - C[NMAX] * np.cos(THETA * NMAX)) + BETA * CE + GAMMA * CO) * DT
    return freq, CHAT

def _FILON_PARAMS(THETA: np.ndarray) -> np.ndarray:
    """
    Calculate the filon parameters.
    """
    SINTH = np.sin(THETA)
    COSTH = np.cos(THETA)
    SINSQ = np.square(SINTH)
    COSSQ = np.square(COSTH)
    THSQ  = np.square(THETA)
    THCUB = THSQ * THETA
    ALPHA = 1. * ( THSQ + THETA * SINTH * COSTH - 2. * SINSQ )
    BETA  = 2. * ( THETA * ( 1. + COSSQ ) - 2. * SINTH * COSTH )
    GAMMA = 4. * ( SINTH - THETA * COSTH )
    ALPHA[0] = 0.
    BETA[0] = 2. / 3.
    GAMMA[0] = 4. / 3.
    ALPHA[1:] /= THCUB[1:]
    BETA[1:] /= THCUB[1:]
    GAMMA[1:] /= THCUB[1:]
    return ALPHA, BETA, GAMMA

def _FFT_OE(C: np.ndarray, M: int):
    M0 = int(M / 2)
    DTH = 2 * np.pi / M

    # Even coordinates
    CE = _range_fft(C[::2], M0) # type: ignore
    CE = np.concatenate([CE, CE, CE[0:1]])

    # Odd coordinates
    CO = _range_fft(C[1::2], M0) * np.exp(-np.arange(M0) * DTH * 1j) # type: ignore
    CO = np.concatenate([CO, -CO, CO[0:1]])
    return CE, CO

def _range_fft(a: np.ndarray, n: Optional[int] = None, axis: int = -1):
    """
    Compute `a_hat[..., l, ...] = \sum_{k=1}^{a.shape[axis]} a[..., k, ...]e^{-(2kl\pi/n)}`
    """
    axis %= a.ndim
    l = a.shape[axis]
    if n is None:
        n = l
    if n >= l:
        return np.fft.fft(a, n, axis)
    num_n = int(l / n)
    l0 = n * num_n
    new_shape = list(a.shape)
    new_shape[axis] = n
    new_shape.insert(axis, num_n)
    a_main = np.sum(a.take(range(l0), axis).reshape(new_shape), axis)
    a_tail = a.take(range(l0, l), axis)
    return np.fft.fft(a_main, n, axis) + np.fft.fft(a_tail, n, axis)

def change_unit_ir(freq_ps, CHAT: np.ndarray, temperature: float):
    a0 = 0.52917721067e-10  # m
    cc = 2.99792458e8;      # m/s
    kB = 1.38064852*1.0e-23 # J/K
    beta = 1.0 / (kB * temperature); 
	# 1 Debye = 0.20819434 e*Angstrom
	# 1 e = 1.602*1.0e-19 C
	# change unit to C*m for M(0)
    unit_basic = 1.602176565 * 1.0e-19 * a0
	# change unit to ps for dM(0)/dt
    unitt = unit_basic / 1
	# because dot(M(0))*dot(M(t)) change unit to C^2 * m^2 / ps^2
    unit2 = unitt**2
    epsilon0 = 8.8541878e-12 # F/m = C^2 / (J * m)
    unit_all = beta / (3.0 * cc * a0 ** 3) / (2 * epsilon0) * unit2
    unit_all = unit_all * 1.0e12 * 1.0e-2; # ps to s, m-1 to cm-1
    CHAT *= unit_all
    d_omega = freq_ps / cc     # Wavenumber
    d_omega *= 1e10         # cm^-1
    return d_omega, CHAT

def change_unit_raman(freq_ps, CHAT: np.ndarray, temperature: float):
    cc = 2.99792458e8;                  # m/s
    kB = 1.38064852*1.0e-23             # J/K
    h = 6.62607015e-34                  # J*s
    h_bar = h / (2 * np.pi)
    beta = 1.0 / (kB * temperature);    # J^-1
    freq = 2 * np.pi * freq_ps * 1e12   # s^-1
    CHAT = CHAT * 1e4 * (1 - np.exp(-beta * h_bar * freq * np.arange(CHAT.shape[0])))
    d_omega = 1e10 * freq_ps / cc       # cm^-1
    return d_omega, CHAT

def change_unit_sfg(freq_ps, CHAT: np.ndarray, temperature: float):
    a0 = 0.52917721067e-10  # m
    cc = 2.99792458e8;      # m/s
    kB = 1.38064852*1.0e-23 # J/K
    beta = 1.0 / (kB * temperature); 
	# 1 Debye = 0.20819434 e*Angstrom
	# 1 e = 1.602*1.0e-19 C
	# change unit to C*m for M(0)
    unit_basic = 1.602176565 * 1.0e-19 * a0
	# change unit to ps for dM(0)/dt
    unitt = unit_basic / 1
	# because dot(M(0))*dot(M(t)) change unit to C^2 * m^2 / ps^2
    unit2 = unitt**2
    epsilon0 = 8.8541878e-12 # F/m = C^2 / (J * m)
    unit_all = beta / (4 * np.pi * a0 ** 2) / (2 * epsilon0) * unit2
    unit_all = unit_all * 1.0e12 * 1.0e-5; # ps to s, m-1to 1000cm-1
    CHAT *= unit_all * freq_ps * 1e4 * np.arange(CHAT.shape[0])
    d_omega = 1e10 * freq_ps / cc
    return d_omega, CHAT

def calculate_ir_atomic(corr: np.ndarray, width: float, dt_ps: float, temperature: float, 
                 M: Optional[int] = None, filter_type: str = "gaussian"):
    nmax = corr.shape[0] - 1
    if nmax % 2 != 0:
        nmax -= 1
        corr = corr[:-1]
    tmax = nmax * dt_ps
    filter_type = filter_type.lower().strip()
    print("nmax         =", nmax)
    print("dt   (ps)    =", dt_ps)
    print("tmax (ps)    =", tmax)
    print("Filter type  =", filter_type)
    print("Smooth width =", width)
    if filter_type == "gaussian":
        width = width * tmax / 100.0 * 3
        C = apply_gussian_filter(corr, width)
    elif filter_type == "lorenz":
        C = apply_lorenz_filter(corr, width, dt_ps)
    else:
        raise NotImplementedError(f"Unknown filter type: {filter_type}!")
    freq_ps, CHAT = FT(dt_ps, C, M)
    d_omega, CHAT = change_unit_ir(freq_ps, CHAT, temperature)
    return np.arange(CHAT.shape[0]) * d_omega, CHAT

def calculate_raman_atomic(corr: np.ndarray, width: float, dt_ps: float, temperature: float, 
                    M: Optional[int] = None, filter_type: str = "gaussian"):
    nmax = corr.shape[0] - 1
    if nmax % 2 != 0:
        nmax -= 1
        corr = corr[:-1]
    tmax = nmax * dt_ps        # ps
    filter_type = filter_type.lower().strip()
    print('nmax         =', nmax)
    print('dt   (ps)    =', dt_ps)
    print('tmax (ps)    =', tmax)
    print("Filter type  =", filter_type)
    print("width        = ", width)
    if filter_type == "gaussian":
        width = width * tmax / 100.0 * 3
        C = apply_gussian_filter(corr, width)
    elif filter_type == "lorenz":
        C = apply_lorenz_filter(corr, width, dt_ps)
    else:
        raise NotImplementedError(f"Unknown filter type: {filter_type}!")
    freq_ps, CHAT = FT(dt_ps, C, M)
    d_omega, CHAT = change_unit_raman(freq_ps, CHAT, temperature)
    return np.arange(CHAT.shape[0]) * d_omega, CHAT

def calculate_sfg(corr: np.ndarray, width: int, dt_ps: float, temperature: float, 
                  M: Optional[int] = None, filter_type: str = "gaussian"):
    nmax = corr.shape[0] - 1
    if nmax % 2 != 0:
        nmax -= 1
        corr = corr[:-1]
    tmax = nmax * dt_ps
    filter_type = filter_type.lower().strip()
    print('nmax         =', nmax)
    print('dt   (ps)    =', dt_ps)
    print('tmax (ps)    =', tmax)
    print("Filter type  =", filter_type)
    print("width        = ", width)
    if filter_type == "gaussian":
        width = width * tmax / 100.0 * 3
        C = apply_gussian_filter(corr, width)
    elif filter_type == "lorenz":
        C = apply_lorenz_filter(corr, width, dt_ps)
    else:
        raise NotImplementedError(f"Unknown filter type: {filter_type}!")
    freq_ps, CHAT_COS = FT    (dt_ps, C, M)
    _      , CHAT_SIN = FT_sin(dt_ps, C, M)
    d_omega, CHAT_COS = change_unit_sfg(freq_ps, CHAT_COS, temperature)
    _      , CHAT_SIN = change_unit_sfg(freq_ps, CHAT_SIN, temperature)
    return np.arange(CHAT_COS.shape[0]) * d_omega, -CHAT_COS, CHAT_SIN