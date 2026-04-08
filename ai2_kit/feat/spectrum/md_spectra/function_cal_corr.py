from typing import Optional
import numpy as np
from .function_prepare import get_distance

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

def calculate_corr_vdipole(atomic_dipole: np.ndarray, weight: np.ndarray, coords: np.ndarray, 
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

def calculate_corr_polar(atomic_polar: np.ndarray, weight: np.ndarray, coords: np.ndarray, 
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