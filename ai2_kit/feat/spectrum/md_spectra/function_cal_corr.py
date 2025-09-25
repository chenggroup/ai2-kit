from typing import Optional
import numpy as np

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

def calculate_corr_vdipole(dipole: np.ndarray, dt_ps: float, window: int):
    v_dipole = (dipole[2:] - dipole[:-2]) / (2 * dt_ps)
    v_dipole -= np.mean(v_dipole, axis = 0, keepdims = True)
    corr = np.sum(calculate_corr(v_dipole, v_dipole, window), axis = -1)
    return corr

def calculate_corr_polar(polar: np.ndarray, window: int):
    polar_iso = np.mean(polar.diagonal(offset = 0, axis1 = 1, axis2 = 2), axis = 1)

    diag = np.zeros((polar_iso.shape[0], 3, 3), dtype = float)
    diag[:, 0, 0] = polar_iso
    diag[:, 1, 1] = polar_iso
    diag[:, 2, 2] = polar_iso
    polar_aniso = polar - diag

    polar_iso -= np.mean(polar_iso, axis = 0, keepdims = True)
    polar_aniso -= np.mean(polar_aniso, axis = 0, keepdims = True)

    corr_iso = calculate_corr(polar_iso, polar_iso, window)
    polar_aniso = polar_aniso.reshape(-1, 9)
    corr_aniso = calculate_corr(polar_aniso, polar_aniso, window)
    corr_aniso *= 2 / 15
    return corr_iso, corr_aniso

def cutoff_sfg(z_arr, z0, zc, zw):
    cut_f = lambda x: np.cos(np.pi * np.clip(x, -1, 0) / 2)**2
    z = z_arr - z0
    return np.sign(z) * cut_f((np.abs(z) - (zc + zw)) / zw)

def cal_weighted_dipole(z, atomic_dipole, z0, zc, zw):
    weight = cutoff_sfg(z, z0, zc, zw)
    weighted_dipole = weight * atomic_dipole / np.sqrt(np.clip(np.sum(np.abs(weight), axis = 1, keepdims = True), 1, None))
    return weighted_dipole

def cal_weighted_polar(z, atomic_dipole, z0, zc, zw):
    weight = cutoff_sfg(z, z0, zc, zw)
    weighted_polar = (weight ** 2) * atomic_dipole / np.sqrt(np.clip(np.sum(weight ** 2, axis = 1, keepdims = True), 1, None))
    return weighted_polar

def cal_corr_sfg(atomic_polar: np.ndarray, atomic_dipole: np.ndarray, coords: np.ndarray, window: int, rc: float = 6.75):
    atomic_polar -= np.mean(atomic_polar, axis = 0, keepdims = True)
    atomic_dipole -= np.mean(atomic_dipole, axis = 0, keepdims = True)
    natom = atomic_dipole.shape[1]
    dipole_cutoff = np.empty_like(atomic_dipole)
    for atom_i in range(natom):
        dis_mask = np.linalg.norm(coords - coords[:, [atom_i], :], ord=2, axis=-1) < rc
        dipole_cutoff[:, atom_i] = np.sum(atomic_dipole * dis_mask, axis=1)
    corr = np.sum(calculate_corr(dipole_cutoff, atomic_polar, window), axis = -1)
    return corr



