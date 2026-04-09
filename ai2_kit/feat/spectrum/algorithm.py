import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import dpdata
from ase import Atoms
from MDAnalysis.lib.distances import distance_array, minimize_vectors

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


def _coords_cells_mul(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    if coords.ndim >= cells.ndim:
        d0 = coords.ndim - cells.ndim + 1
        shape = coords.shape
        return np.matmul(coords.reshape(shape[:-d0 - 1] + (-1, 3)), cells).reshape(shape)
    return np.matmul(coords[..., None, :], cells).squeeze(-2)


def inv_cells(cells: np.ndarray):
    return np.linalg.inv(cells)


def to_frac(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    return _coords_cells_mul(coords, inv_cells(cells))


def box_shift(dx: np.ndarray, cells: np.ndarray) -> np.ndarray:
    return dx - _coords_cells_mul(np.round(to_frac(dx, cells)), cells)


def do_pbc(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    return coords - _coords_cells_mul(np.floor(to_frac(coords, cells)), cells)


def get_distance(
    coords_A: np.ndarray,
    coords_B: Optional[np.ndarray],
    cells: np.ndarray,
    remove_diag: bool = False,
    offset: int = 0,
):
    if coords_B is None:
        coords_B = coords_A
    distance = np.linalg.norm(
        box_shift(coords_A[..., None, :] - coords_B[..., None, :, :], cells),
        ord=2,
        axis=-1,
    )
    if remove_diag:
        write_to_diagonal(distance, np.inf, offset=offset, axis1=-2, axis2=-1)
    return distance


def k_nearest(coords_A: np.ndarray, coords_B: Optional[np.ndarray], cells: np.ndarray, k: int, batch_size: int = -1):
    self_comp = False
    if coords_B is None:
        coords_B = coords_A
        self_comp = True
    d = coords_B.shape[-2]
    k = min(d, k)
    batch_size = min(d - k, batch_size)
    if batch_size <= 0:
        distance = get_distance(coords_A, coords_B, cells, remove_diag=self_comp)
        k_index = np.argpartition(distance, k, axis=-1)[..., :k]
        k_distance = np.take_along_axis(distance, k_index, axis=-1)
    else:
        shape = list(coords_A.shape)
        shape[-1] = k + batch_size
        k_index = np.empty(shape, dtype=int)
        k_distance = np.empty(shape, dtype=coords_B.dtype)
        k_index[..., :k] = np.arange(k)
        k_distance[..., :k] = get_distance(coords_A, coords_B[..., :k, :], cells, remove_diag=self_comp, offset=0)
        for i in range(k, d, batch_size):
            end_i = min(d, i + batch_size)
            sz = end_i - i
            k_index[..., k:k + sz] = np.arange(i, end_i)
            k_distance[..., k:k + sz] = get_distance(
                coords_A, coords_B[..., i:end_i, :], cells, remove_diag=self_comp, offset=i
            )
            partition_idx = np.argpartition(k_distance, k, axis=-1)
            k_index = np.take_along_axis(k_index, partition_idx, axis=-1)
            k_distance = np.take_along_axis(k_distance, partition_idx, axis=-1)
    sort_idx = np.argsort(k_distance[..., :k], axis=-1)
    return np.take_along_axis(k_index[..., :k], sort_idx, axis=-1)


def find_h2o(coords_sel: np.ndarray, coords_oth: np.ndarray, cells: np.ndarray, r_bond: float):
    coords_sel = coords_sel[..., np.newaxis, :]
    delta = box_shift(coords_oth[..., np.newaxis, :, :] - coords_sel, cells[..., np.newaxis, np.newaxis, :, :])  # type: ignore
    mask = np.linalg.norm(delta, 2, axis=-1) < r_bond
    return np.sum(mask, axis=-1) == 2


def calculate_dipole_H(coords_O: np.ndarray, coords_H: np.ndarray, cells: np.ndarray):
    idx_H = k_nearest(coords_O[[0]], coords_H[[0]], cells[[0]], 2)
    cH = np.take_along_axis(coords_H[..., np.newaxis, :, :], idx_H[..., np.newaxis], axis=-2)
    return np.sum(box_shift(cH - coords_O[..., np.newaxis, :], cells[..., np.newaxis, np.newaxis, :, :]), axis=-2)


def calculate_dipole(coords_O: np.ndarray, coords_H: np.ndarray, cells: np.ndarray, wannier: np.ndarray) -> np.ndarray:
    return calculate_dipole_H(coords_O, coords_H, cells) - wannier * 8


def calculate_dipole_OH_H(coords_O: np.ndarray, coords_H: np.ndarray, coords_Al: np.ndarray, cells: np.ndarray):
    idx_H = k_nearest(coords_O[[0]], coords_H[[0]], cells[[0]], 1)
    idx_Al = k_nearest(coords_O[[0]], coords_Al[[0]], cells[[0]], 2)
    cH = np.take_along_axis(coords_H[..., np.newaxis, :, :], idx_H[..., np.newaxis], axis=-2)
    cAl = np.take_along_axis(coords_Al[..., np.newaxis, :, :], idx_Al[..., np.newaxis], axis=-2)
    return (
        np.sum(box_shift(cH - coords_O[..., np.newaxis, :], cells[..., np.newaxis, np.newaxis, :, :]), axis=-2)
        + np.sum(box_shift(cAl - coords_O[..., np.newaxis, :], cells[..., np.newaxis, np.newaxis, :, :]), axis=-2) / 6
    )


def calculate_dipole_OH(
    coords_O: np.ndarray,
    coords_H: np.ndarray,
    coords_Al: np.ndarray,
    cells: np.ndarray,
    wannier: np.ndarray,
) -> np.ndarray:
    return calculate_dipole_OH_H(coords_O, coords_H, coords_Al, cells) - wannier * 8


def calculate_corr(A: np.ndarray, B: np.ndarray, NMAX: int, window: Optional[int] = None):
    if A.ndim == 1 or B.ndim == 1:
        A = A.reshape(-1, 1)
        B = B.reshape(-1, 1)
    if window is None:
        window = min(A.shape[0], B.shape[0] - NMAX)
    v1 = A[:window][::-1]
    v2 = B[:window + NMAX]
    pad_width = [(0, 0)] * A.ndim
    pad_width[0] = (0, NMAX)
    v1 = np.pad(v1, pad_width, "constant", constant_values=0)
    corr = np.fft.ifft(np.fft.fft(v1, axis=0) * np.fft.fft(v2, axis=0), axis=0).real
    return corr[window - 1:window + NMAX] / window


def cutoff_ir_raman(arr, low, high, smooth_width):
    eps = 1e-2
    cut_f = lambda x: np.exp(-1 / np.clip(x, eps, None))
    a_in = cut_f(np.maximum(low - arr, arr - high))
    a_out = cut_f(np.minimum(arr - low, high - arr) + smooth_width)
    return a_out / (a_out + a_in)


def cal_range_dipole_polar(z, atomic_dipole, z_lo, z_hi, r_smth):
    weight = cutoff_ir_raman(z, z_lo, z_hi, r_smth)
    return np.sum(weight * atomic_dipole, axis=1) / np.sqrt(np.clip(np.sum(weight, axis=1), 1, None))


def calculate_corr_vdipole(
    atomic_dipole: np.ndarray,
    weight: np.ndarray,
    coords: np.ndarray,
    cells: np.ndarray,
    dt_ps: float,
    window: int,
    rc: float = 6.75,
):
    natom = atomic_dipole.shape[1]
    weight = weight[1:-1]
    coords = coords[1:-1]
    cells = cells[1:-1]
    weight /= np.sqrt(np.clip(np.sum(np.abs(weight), axis=1, keepdims=True), 1, None))
    v_dipole = weight[..., None] * (atomic_dipole[2:] - atomic_dipole[:-2]) / (2 * dt_ps)
    corr_intra = calculate_corr(v_dipole, v_dipole, window)
    dipole_cutoff = np.empty_like(v_dipole)
    for atom_i in range(natom):
        dis_mask = get_distance(coords, coords[:, [atom_i], :], cells) < rc
        dis_mask[:, atom_i] = False
        dipole_cutoff[:, atom_i] = np.matmul(v_dipole.transpose(0, 2, 1), dis_mask).squeeze(2)
    corr_inter = calculate_corr(dipole_cutoff, v_dipole, window)
    return corr_intra, corr_inter


def calculate_corr_polar(
    atomic_polar: np.ndarray,
    weight: np.ndarray,
    coords: np.ndarray,
    cells: np.ndarray,
    window: int,
    rc: float = 6.75,
):
    nframes, natom = atomic_polar.shape[:2]
    polar_iso = np.mean(atomic_polar.diagonal(offset=0, axis1=-2, axis2=-1), axis=-1)
    diag = np.zeros_like(atomic_polar, dtype=float)
    diag[..., 0, 0] = polar_iso
    diag[..., 1, 1] = polar_iso
    diag[..., 2, 2] = polar_iso
    polar_aniso = atomic_polar - diag
    polar_iso -= np.mean(polar_iso, axis=0, keepdims=True)
    polar_aniso -= np.mean(polar_aniso, axis=0, keepdims=True)
    polar_iso = np.square(weight) * polar_iso
    polar_aniso = np.square(weight[..., None, None]) * polar_aniso
    polar_aniso = polar_aniso.reshape(nframes, natom, 9)
    corr_iso_intra = calculate_corr(polar_iso, polar_iso, window)
    corr_aniso_intra = np.sum(calculate_corr(polar_aniso, polar_aniso, window), axis=-1) * (2.0 / 15.0)
    polar_iso_cutoff = np.empty_like(polar_iso)
    polar_aniso_cutoff = np.empty_like(polar_aniso)
    for atom_i in range(natom):
        dis_mask = get_distance(coords, coords[:, [atom_i], :], cells) < rc
        dis_mask[:, atom_i] = False
        polar_iso_cutoff[:, atom_i] = np.matmul(polar_iso[:, None, :], dis_mask).squeeze((1, 2))
        polar_aniso_cutoff[:, atom_i] = np.matmul(polar_aniso.transpose(0, 2, 1), dis_mask).squeeze(2)
    corr_iso_inter = calculate_corr(polar_iso_cutoff, polar_iso, window)
    corr_aniso_inter = np.sum(calculate_corr(polar_aniso_cutoff, polar_aniso, window), axis=-1) * (2.0 / 15.0)
    return corr_iso_intra, corr_aniso_intra, corr_iso_inter, corr_aniso_inter


def cutoff_z(z_arr, z0, zc, zw):
    cut_f = lambda x: np.cos(np.pi * np.clip(x, -1, 0) / 2) ** 2
    z = z_arr - z0
    return np.sign(z) * cut_f((np.abs(z) - (zc + zw)) / zw)


def cal_corr_sfg_method2(
    atomic_polar: np.ndarray,
    atomic_dipole: np.ndarray,
    weight: np.ndarray,
    coords: np.ndarray,
    cells: np.ndarray,
    window: int,
    rc: float = 6.75,
):
    natom = atomic_dipole.shape[1]
    atomic_polar -= np.mean(atomic_polar, axis=0, keepdims=True)
    atomic_dipole -= np.mean(atomic_dipole, axis=0, keepdims=True)
    dipole = weight * atomic_dipole
    corr_intra = calculate_corr(dipole, np.square(weight) * atomic_polar, window)
    dipole_cutoff = np.empty_like(dipole)
    for atom_i in range(natom):
        dis_mask = get_distance(coords, coords[:, [atom_i], :], cells) < rc
        dis_mask[:, atom_i] = False
        dipole_cutoff[:, atom_i] = np.matmul(dipole[:, None, :], dis_mask).squeeze() * np.square(weight[:, atom_i])
    corr_inter = calculate_corr(dipole_cutoff, atomic_polar, window)
    return corr_intra, corr_inter


def cal_corr_sfg_method1(
    atomic_polar: np.ndarray,
    atomic_dipole: np.ndarray,
    weight: np.ndarray,
    coords: np.ndarray,
    cells: np.ndarray,
    window: int,
    rc: float = 6.75,
):
    natom = atomic_dipole.shape[1]
    atomic_polar -= np.mean(atomic_polar, axis=0, keepdims=True)
    atomic_dipole -= np.mean(atomic_dipole, axis=0, keepdims=True)
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


calculate_corr_vdipole_atomic = calculate_corr_vdipole
calculate_corr_polar_atomic = calculate_corr_polar


def apply_gussian_filter(corr: np.ndarray, width: float):
    nmax = corr.shape[0] - 1
    return corr * np.exp(-0.5 * (0.5 * width * np.arange(nmax + 1) / nmax) ** 2)


def apply_lorenz_filter(corr: np.ndarray, width: float, dt):
    nmax = corr.shape[0] - 1
    b = width * 2.99792458e-3
    M = int(1 / (dt * 0.01 * 2)) * 2
    M = max(M, nmax)
    dx = 1 / (M * dt)
    NX = int(50 * np.sqrt(b) / dx / 2) * 2
    x = np.arange(NX + 1) * dx
    p = b / (b ** 2 + x ** 2)
    _, ph = FT(dx, p, M)
    return corr * ph[:nmax + 1]


def _range_fft(a: np.ndarray, n: Optional[int] = None, axis: int = -1):
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


def _FFT_OE(C: np.ndarray, M: int):
    M0 = int(M / 2)
    DTH = 2 * np.pi / M
    CE = _range_fft(C[::2], M0)
    CE = np.concatenate([CE, CE, CE[0:1]])
    CO = _range_fft(C[1::2], M0) * np.exp(-np.arange(M0) * DTH * 1j)
    CO = np.concatenate([CO, -CO, CO[0:1]])
    return CE, CO


def _FILON_PARAMS(THETA: np.ndarray) -> np.ndarray:
    SINTH = np.sin(THETA)
    COSTH = np.cos(THETA)
    SINSQ = np.square(SINTH)
    COSSQ = np.square(COSTH)
    THSQ = np.square(THETA)
    THCUB = THSQ * THETA
    ALPHA = 1.0 * (THSQ + THETA * SINTH * COSTH - 2.0 * SINSQ)
    BETA = 2.0 * (THETA * (1.0 + COSSQ) - 2.0 * SINTH * COSTH)
    GAMMA = 4.0 * (SINTH - THETA * COSTH)
    ALPHA[0] = 0.0
    BETA[0] = 2.0 / 3.0
    GAMMA[0] = 4.0 / 3.0
    ALPHA[1:] /= THCUB[1:]
    BETA[1:] /= THCUB[1:]
    GAMMA[1:] /= THCUB[1:]
    return ALPHA, BETA, GAMMA


def FT(DT: float, C: np.ndarray, M: Optional[int] = None) -> np.ndarray:
    NMAX = C.shape[0] - 1
    if NMAX % 2 != 0:
        raise ValueError("NMAX (=len(C)-1) must be even for the cosine transform.")
    if M is None:
        M = NMAX
    elif M % 2 != 0:
        M += 1
    freq = 1 / (M * DT)
    DTH = 2 * np.pi / M
    THETA = np.arange(M + 1) * DTH
    ALPHA, BETA, GAMMA = _FILON_PARAMS(THETA)
    CE, CO = _FFT_OE(C, M)
    CE, CO = CE.real, CO.real
    CE -= 0.5 * (C[0] + C[NMAX] * np.cos(THETA * NMAX))
    CHAT = 2.0 * (ALPHA * C[NMAX] * np.sin(THETA * NMAX) + BETA * CE + GAMMA * CO) * DT
    return freq, CHAT


def FT_sin(DT: float, C: np.ndarray, M: Optional[int] = None) -> np.ndarray:
    NMAX = C.shape[0] - 1
    if NMAX % 2 != 0:
        raise ValueError("NMAX (=len(C)-1) must be even for the sine transform.")
    if M is None:
        M = NMAX
    elif M % 2 != 0:
        M += 1
    freq = 1 / (M * DT)
    DTH = 2 * np.pi / M
    THETA = np.arange(M + 1) * DTH
    ALPHA, BETA, GAMMA = _FILON_PARAMS(THETA)
    CE, CO = _FFT_OE(C, M)
    CE, CO = CE.imag, CO.imag
    CE -= 0.5 * (C[NMAX] * np.sin(THETA * NMAX))
    CHAT = 2.0 * (ALPHA * (C[0] - C[NMAX] * np.cos(THETA * NMAX)) + BETA * CE + GAMMA * CO) * DT
    return freq, CHAT


def change_unit_ir(freq_ps, CHAT: np.ndarray, temperature: float):
    a0 = 0.52917721067e-10
    cc = 2.99792458e8
    kB = 1.38064852 * 1.0e-23
    beta = 1.0 / (kB * temperature)
    unit_basic = 1.602176565 * 1.0e-19 * a0
    unitt = unit_basic / 1
    unit2 = unitt ** 2
    epsilon0 = 8.8541878e-12
    unit_all = beta / (3.0 * cc * a0 ** 3) / (2 * epsilon0) * unit2
    unit_all = unit_all * 1.0e12 * 1.0e-2
    CHAT *= unit_all
    d_omega = 1e10 * freq_ps / cc
    return d_omega, CHAT


def change_unit_raman(freq_ps, CHAT: np.ndarray, temperature: float):
    cc = 2.99792458e8
    kB = 1.38064852 * 1.0e-23
    h = 6.62607015e-34
    h_bar = h / (2 * np.pi)
    beta = 1.0 / (kB * temperature)
    freq = 2 * np.pi * freq_ps * 1e12
    CHAT = CHAT * 1e4 * (1 - np.exp(-beta * h_bar * freq * np.arange(CHAT.shape[0])))
    d_omega = 1e10 * freq_ps / cc
    return d_omega, CHAT


def change_unit_sfg(freq_ps, CHAT: np.ndarray, temperature: float):
    a0 = 0.52917721067e-10
    cc = 2.99792458e8
    kB = 1.38064852 * 1.0e-23
    beta = 1.0 / (kB * temperature)
    unit_basic = 1.602176565 * 1.0e-19 * a0
    unitt = unit_basic / 1
    unit2 = unitt ** 2
    epsilon0 = 8.8541878e-12
    unit_all = beta / (4 * np.pi * a0 ** 2) / (2 * epsilon0) * unit2
    unit_all = unit_all * 1.0e12 * 1.0e-5
    CHAT *= unit_all * freq_ps * 1e4 * np.arange(CHAT.shape[0])
    d_omega = 1e10 * freq_ps / cc
    return d_omega, CHAT


def calculate_ir(corr: np.ndarray, width: float, dt_ps: float, temperature: float, M: Optional[int] = None, filter_type: str = "gaussian"):
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
        C = apply_gussian_filter(corr, width * tmax / 100.0 * 3)
    elif filter_type == "lorenz":
        C = apply_lorenz_filter(corr, width, dt_ps)
    else:
        raise NotImplementedError(f"Unknown filter type: {filter_type}!")
    freq_ps, CHAT = FT(dt_ps, C, M)
    d_omega, CHAT = change_unit_ir(freq_ps, CHAT, temperature)
    return np.arange(CHAT.shape[0]) * d_omega, CHAT


def calculate_raman(corr: np.ndarray, width: float, dt_ps: float, temperature: float, M: Optional[int] = None, filter_type: str = "gaussian"):
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
    print("width        = ", width)
    if filter_type == "gaussian":
        C = apply_gussian_filter(corr, width * tmax / 100.0 * 3)
    elif filter_type == "lorenz":
        C = apply_lorenz_filter(corr, width, dt_ps)
    else:
        raise NotImplementedError(f"Unknown filter type: {filter_type}!")
    freq_ps, CHAT = FT(dt_ps, C, M)
    d_omega, CHAT = change_unit_raman(freq_ps, CHAT, temperature)
    return np.arange(CHAT.shape[0]) * d_omega, CHAT


def calculate_sfg(corr: np.ndarray, width: int, dt_ps: float, temperature: float, M: Optional[int] = None, filter_type: str = "gaussian"):
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
    print("width        = ", width)
    if filter_type == "gaussian":
        C = apply_gussian_filter(corr, width * tmax / 100.0 * 3)
    elif filter_type == "lorenz":
        C = apply_lorenz_filter(corr, width, dt_ps)
    else:
        raise NotImplementedError(f"Unknown filter type: {filter_type}!")
    freq_ps, CHAT_COS = FT(dt_ps, C, M)
    _, CHAT_SIN = FT_sin(dt_ps, C, M)
    d_omega, CHAT_COS = change_unit_sfg(freq_ps, CHAT_COS, temperature)
    _, CHAT_SIN = change_unit_sfg(freq_ps, CHAT_SIN, temperature)
    return np.arange(CHAT_COS.shape[0]) * d_omega, -CHAT_COS, CHAT_SIN


calculate_ir_atomic = calculate_ir
calculate_raman_atomic = calculate_raman


def _normalize_for_plot(wavenumber: np.ndarray, intensity: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """
    Helper function to normalize the intensity array based on its maximum absolute value
    within a specific wavenumber window [x_min, x_max].
    
    Args:
        wavenumber (np.ndarray): The x-axis array (frequencies).
        intensity (np.ndarray): The y-axis array (intensities).
        x_min (float): Lower bound of the x-axis for plotting.
        x_max (float): Upper bound of the x-axis for plotting.

    Returns:
        np.ndarray: Intensity (a.u.) array for plotting.
    """
    mask = (wavenumber >= x_min) & (wavenumber <= x_max)
    if np.any(mask):
        max_val = np.max(np.abs(intensity[mask]))
        if max_val == 0:
            max_val = 1.0
    else:
        max_val = 1.0
    return intensity / max_val


def compute_atomic_dipole_h2o(
    traj: "dpdata.System",
    wannier: np.ndarray,
    type_O: int = 0,
    type_H: int = 1,
    r_bond: float = 1.2,
    save_datas: Optional[List[str]] = None,
):
    """
    Compute atomic dipole moments for water molecules from trajectory and Wannier center data.

    Args:
        traj (dpdata.System): System trajectory containing coordinates, cells, and atom types.
        wannier (np.ndarray): Predicted Wannier center coordinates array.
        type_O (int, optional): Atom type index for Oxygen. Defaults to 0.
        type_H (int, optional): Atom type index for Hydrogen. Defaults to 1.
        r_bond (float, optional): Cutoff distance to define an O-H bond. Defaults to 1.2.
        save_datas (Optional[List[str]], optional): List of file paths to save [h2o coordinates, atomic dipoles]. 
                                                    Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - h2o_O: Coordinates of the Oxygen atoms of water molecules.
            - atomic_dipole: The computed atomic dipole array for each water molecule.
    """
    coords = traj["coords"]
    cells = traj["cells"]
    types = traj["atom_types"]
    coords = do_pbc(coords, cells)
    coords_O = coords[:, types == type_O, :]
    coords_H = coords[:, types == type_H, :]

    h2o_mask = find_h2o(coords_O[0], coords_H[0], cells[0], r_bond=r_bond)
    h2o_O = coords_O[:, h2o_mask, :]

    if save_datas is not None and len(save_datas) > 0 and save_datas[0]:
        np.save(save_datas[0], h2o_O)

    wannier = wannier.reshape(traj.get_nframes(), -1, 3)
    wannier_h2o = wannier[:, h2o_mask, :]
    atomic_dipole = calculate_dipole(h2o_O, coords_H, cells, wannier_h2o)

    if save_datas is not None and len(save_datas) > 1 and save_datas[1]:
        np.save(save_datas[1], atomic_dipole.astype(np.float32))

    return h2o_O, atomic_dipole


def extract_atomic_polar_from_traj_h2o(
    traj: "dpdata.System",
    polar: np.ndarray,
    type_O: int = 0,
    type_H: int = 1,
    r_bond: float = 1.2,
    save_data: Optional[str] = None,
):
    """
    Extract atomic polarizability tensors specifically for water molecules.

    Args:
        traj (dpdata.System): System trajectory containing coordinates, cells, and atom types.
        polar (np.ndarray): Raw polarizability tensor array for all components.
        type_O (int, optional): Atom type index for Oxygen. Defaults to 0.
        type_H (int, optional): Atom type index for Hydrogen. Defaults to 1.
        r_bond (float, optional): Cutoff distance to define an O-H bond. Defaults to 1.2.
        save_data (Optional[str], optional): File path to save the water polarizability array. Defaults to None.

    Returns:
        np.ndarray: The extracted atomic polarizability array for water molecules.
    """
    coords = traj["coords"]
    cells = traj["cells"]
    types = traj["atom_types"]

    coords = do_pbc(coords, cells)
    coords_O = coords[:, types == type_O, :]
    coords_H = coords[:, types == type_H, :]
    
    polar = -polar.reshape(polar.shape[0], -1, 3, 3)
    h2o_mask = find_h2o(coords_O[0], coords_H[0], cells[0], r_bond=r_bond)
    polar_h2o = polar[:, h2o_mask, :, :]

    if save_data is not None:
        np.save(save_data, polar_h2o.astype(np.float32))

    return polar_h2o


def compute_bulk_ir_h2o(
    h2o: np.ndarray,
    cells: np.ndarray,
    atomic_dipole: np.ndarray,
    dt: float = 0.0005,
    window: int = 2000,
    z0: float = 25.0,
    zc: float = 7.5,
    zw: float = 0.5,
    rc: float = 6.0,
    width: int = 240,
    temperature: float = 330.0,
    M: int = 20000,
    filter_type: str = "lorenz",
    nuclear_quantum_factor: float = 0.96,
    save_plot: Optional[str] = None,
    save_data: Optional[str] = None,
):
    """
    Compute and optionally save/plot the bulk IR spectrum from atomic dipole data.

    Args:
        h2o (np.ndarray): Water molecule coordinates.
        cells (np.ndarray): Simulation box array.
        atomic_dipole (np.ndarray): Atomic dipole array.
        dt (float, optional): Time step between frames in ps. Defaults to 0.0005.
        window (int, optional): Window size for correlation/frequency interval. Defaults to 2000.
        z0 (float, optional): Z cutoff center for the weighting function. Defaults to 25.0.
        zc (float, optional): Z cutoff width for the bulk region. Defaults to 7.5.
        zw (float, optional): Z cutoff smoothing parameter. Defaults to 0.5.
        rc (float, optional): Cutoff radius for correlation function. Defaults to 6.0.
        width (int, optional): Smoothing width parameter for the IR spectrum. Defaults to 240.
        temperature (float, optional): Simulation temperature in Kelvin. Defaults to 330.0.
        M (int, optional): Number of points for IR Fourier transform. Defaults to 20000.
        filter_type (str, optional): Filter type ("lorenz" or "gaussian"). Defaults to "lorenz".
        nuclear_quantum_factor (float, optional): Correction factor for nuclear quantum effects. Defaults to 0.96.
        save_plot (Optional[str], optional): Path to save the plot figure. Defaults to None.
        save_data (Optional[str], optional): Path to save the raw array data. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - wavenumber: Array of evaluated frequencies (cm^-1).
            - ir: The computed total IR intensity.
    """
    weight = cutoff_z(h2o[..., 2], z0, zc, zw)
    weight = np.ones((h2o.shape[0], h2o.shape[1])) - weight

    corr_intra, corr_inter = calculate_corr_vdipole(atomic_dipole, weight, h2o, cells, dt, window, rc=rc)
    corr_intra = np.sum(corr_intra, axis=1)
    corr_inter = np.sum(corr_inter, axis=1)

    wavenumber = np.array(
        calculate_ir(np.sum(corr_intra, axis=1), width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type)
    )[0]
    ir = np.array(
        calculate_ir(np.sum(corr_intra, axis=1) + np.sum(corr_inter, axis=1), width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type)
    )[1]

    if save_plot is not None:
        scaled_wavenumber = wavenumber * nuclear_quantum_factor
        xlim_min, xlim_max = 0, 4000
        norm_ir = _normalize_for_plot(scaled_wavenumber, ir, xlim_min, xlim_max)
        
        plt.plot(scaled_wavenumber, norm_ir, label="bulk water", scalex=1.5, scaley=2.2)
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(0, 1)
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("IR spectra")
        plt.savefig(save_plot, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

    if save_data is not None:
        np.savetxt(save_data, np.array([wavenumber, ir]).T)

    return wavenumber, ir


def compute_bulk_raman_h2o(
    h2o: np.ndarray,
    cells: np.ndarray,
    atomic_polar: np.ndarray,
    dt: float = 0.0005,
    window: int = 2000,
    z0: float = 25.0,
    zc: float = 7.5,
    zw: float = 0.5,
    rc: float = 6.0,
    width: int = 240,
    temperature: float = 330.0,
    M: int = 20000,
    filter_type: str = "lorenz",
    nuclear_quantum_factor: float = 0.96,
    save_plots: Optional[List[str]] = None,
    save_data: Optional[str] = None,
):
    """
    Compute and optionally save/plot the bulk Raman spectra (isotropic, anisotropic, and low-frequency).

    Args:
        h2o (np.ndarray): Water molecule coordinates.
        cells (np.ndarray): Simulation box array.
        atomic_polar (np.ndarray): Atomic polarizability tensor array.
        dt (float, optional): Time step between frames in ps. Defaults to 0.0005.
        window (int, optional): Window size for correlation calculation. Defaults to 2000.
        z0 (float, optional): Z cutoff center for the weighting function. Defaults to 25.0.
        zc (float, optional): Z cutoff width for the bulk region. Defaults to 7.5.
        zw (float, optional): Z cutoff smoothing parameter. Defaults to 0.5.
        rc (float, optional): Cutoff radius for correlation function. Defaults to 6.0.
        width (int, optional): Smoothing width parameter for the Raman spectrum. Defaults to 240.
        temperature (float, optional): Simulation temperature in Kelvin. Defaults to 330.0.
        M (int, optional): Number of points for Raman Fourier transform. Defaults to 20000.
        filter_type (str, optional): Filter type ("lorenz" or "gaussian"). Defaults to "lorenz".
        nuclear_quantum_factor (float, optional): Correction factor for nuclear quantum effects. Defaults to 0.96.
        save_plots (Optional[List[str]], optional): List of 3 paths to save iso, aniso, and low-freq plots. Defaults to None.
        save_data (Optional[str], optional): Path to save the raw array data. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - wavenumber: Array of evaluated frequencies (cm^-1).
            - total_iso: Isotropic Raman intensity.
            - total_aniso: Anisotropic Raman intensity.
            - low_range: Low-frequency Raman intensity.
    """
    weight = cutoff_z(h2o[..., 2], z0, zc, zw)
    weight = np.ones((h2o.shape[0], h2o.shape[1])) - weight

    corr_iso_intra, corr_aniso_intra, corr_iso_inter, corr_aniso_inter = calculate_corr_polar(
        atomic_polar, weight, h2o, cells, window, rc=rc
    )
    corr_iso_intra = np.sum(corr_iso_intra, axis=1)
    corr_aniso_intra = np.sum(corr_aniso_intra, axis=1)
    corr_iso_inter = np.sum(corr_iso_inter, axis=1)
    corr_aniso_inter = np.sum(corr_aniso_inter, axis=1)

    wavenumber = np.array(calculate_raman(corr_iso_intra, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type))[0]
    raman_iso_intra = np.array(calculate_raman(corr_iso_intra, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type))[1]
    raman_aniso_intra = np.array(calculate_raman(corr_aniso_intra, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type))[1]
    raman_iso_inter = np.array(calculate_raman(corr_iso_inter, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type))[1]
    raman_aniso_inter = np.array(calculate_raman(corr_aniso_inter, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type))[1]

    total_iso = raman_iso_intra + raman_iso_inter
    total_aniso = raman_aniso_intra + raman_aniso_inter
    low_range = wavenumber * (raman_aniso_intra + raman_aniso_inter) / 1000

    if save_plots is not None and len(save_plots) == 3:
        scaled_wavenumber = wavenumber * nuclear_quantum_factor
        
        # Isotropic Raman
        xlim_min, xlim_max = 2800, 4000
        norm_iso = _normalize_for_plot(scaled_wavenumber, total_iso, xlim_min, xlim_max)
        plt.plot(scaled_wavenumber, norm_iso, label="bulk water", scalex=1.5, scaley=2.2)
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(0, 1)
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("Raman spectra (iso)")
        plt.savefig(save_plots[0], dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

        # Anisotropic Raman
        norm_aniso = _normalize_for_plot(scaled_wavenumber, total_aniso, xlim_min, xlim_max)
        plt.plot(scaled_wavenumber, norm_aniso, label="bulk water", scalex=1.5, scaley=2.2)
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(0, 1)
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("Raman spectra (aniso)")
        plt.savefig(save_plots[1], dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

        # Low-frequency Raman
        xlim_min_low, xlim_max_low = 0, 2500
        norm_low = _normalize_for_plot(scaled_wavenumber, low_range, xlim_min_low, xlim_max_low)
        plt.plot(scaled_wavenumber, norm_low, label="bulk water", scalex=1.5, scaley=2.2)
        plt.xlim(xlim_min_low, xlim_max_low)
        plt.ylim(0, 1)
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("Low-frequency Raman spectra (aniso)")
        plt.savefig(save_plots[2], dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

    if save_data is not None:
        np.savetxt(save_data, np.array([wavenumber, total_iso, total_aniso, low_range]).T)

    return wavenumber, total_iso, total_aniso, low_range


def compute_surface_ir_spectra_h2o(
    h2o: np.ndarray,
    cells: np.ndarray,
    atomic_dipole: np.ndarray,
    dt: float = 0.0005,
    window: int = 2000,
    z0: float = 25.0,
    zc: float = 7.5,
    zw: float = 0.5,
    rc: float = 6.0,
    width: int = 240,
    temperature: float = 330.0,
    M: int = 20000,
    filter_type: str = "lorenz",
    nuclear_quantum_factor: float = 0.96,
    save_plot: Optional[str] = None,
    save_data: Optional[str] = None,
):
    """
    Compute and optionally plot/save surface IR spectra for interfacial water.

    Args:
        h2o (np.ndarray): Interfacial water molecule coordinates.
        cells (np.ndarray): Simulation box array.
        atomic_dipole (np.ndarray): Atomic dipole array.
        dt (float, optional): Time step between frames in ps. Defaults to 0.0005.
        window (int, optional): Window size for correlation function. Defaults to 2000.
        z0 (float, optional): Z cutoff center to isolate surface water. Defaults to 25.0.
        zc (float, optional): Z cutoff width to isolate surface water. Defaults to 7.5.
        zw (float, optional): Z cutoff smoothing parameter. Defaults to 0.5.
        rc (float, optional): Cutoff radius for correlation function. Defaults to 6.0.
        width (int, optional): Smoothing width parameter for IR spectrum. Defaults to 240.
        temperature (float, optional): Simulation temperature in Kelvin. Defaults to 330.0.
        M (int, optional): Number of points for Fourier transform. Defaults to 20000.
        filter_type (str, optional): Filter type ("lorenz" or "gaussian"). Defaults to "lorenz".
        nuclear_quantum_factor (float, optional): Correction factor for nuclear quantum effects. Defaults to 0.96.
        save_plot (Optional[str], optional): Path to save the plot figure. Defaults to None.
        save_data (Optional[str], optional): Path to save the raw array data. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - wavenumber: Array of evaluated frequencies (cm^-1).
            - ir_h2o: The computed IR intensity for surface water.
    """
    weight = cutoff_z(h2o[..., 2], z0, zc, zw)

    corr_intra, corr_inter = calculate_corr_vdipole(atomic_dipole, weight, h2o, cells, dt, window, rc=rc)
    corr_intra_h2o = np.sum(corr_intra[:, :h2o.shape[1]], axis=1)
    corr_inter_h2o = np.sum(corr_inter[:, :h2o.shape[1]], axis=1)

    wavenumber = np.array(
        calculate_ir(np.sum(corr_intra_h2o, axis=1), width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type)
    )[0]
    ir_h2o = np.array(
        calculate_ir(np.sum(corr_intra_h2o + corr_inter_h2o, axis=1), width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type)
    )[1]

    if save_plot is not None:
        scaled_wavenumber = wavenumber * nuclear_quantum_factor
        xlim_min, xlim_max = 0, 4000
        norm_ir = _normalize_for_plot(scaled_wavenumber, ir_h2o, xlim_min, xlim_max)
        
        plt.plot(scaled_wavenumber, norm_ir, label="interfacial water", scalex=1.5, scaley=2.2)
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(0, 1)
        plt.xlabel(r"Wavenumber($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("IR spectra")
        plt.savefig(save_plot, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

    if save_data is not None:
        np.savetxt(save_data, np.array([wavenumber, ir_h2o]).T)

    return wavenumber, ir_h2o


def compute_surface_raman_h2o(
    h2o: np.ndarray,
    cells: np.ndarray,
    atomic_polar: np.ndarray,
    dt: float = 0.0005,
    window: int = 2000,
    z0: float = 25.0,
    zc: float = 7.5,
    zw: float = 0.5,
    rc: float = 6.0,
    width: int = 240,
    temperature: float = 330.0,
    M: int = 20000,
    filter_type: str = "lorenz",
    nuclear_quantum_factor: float = 0.96,
    save_plots: Optional[List[str]] = None,
    save_data: Optional[str] = None,
):
    """
    Compute and optionally save the surface Raman spectra for interfacial water.

    Args:
        h2o (np.ndarray): Interfacial water molecule coordinates.
        cells (np.ndarray): Simulation box array.
        atomic_polar (np.ndarray): Atomic polarizability tensor array.
        dt (float, optional): Time step between frames in ps. Defaults to 0.0005.
        window (int, optional): Window size for correlation calculation. Defaults to 2000.
        z0 (float, optional): Z cutoff center to isolate surface water. Defaults to 25.0.
        zc (float, optional): Z cutoff width to isolate surface water. Defaults to 7.5.
        zw (float, optional): Z cutoff smoothing parameter. Defaults to 0.5.
        rc (float, optional): Cutoff radius for correlation function. Defaults to 6.0.
        width (int, optional): Smoothing width parameter for the Raman spectrum. Defaults to 240.
        temperature (float, optional): Simulation temperature in Kelvin. Defaults to 330.0.
        M (int, optional): Number of points for Fourier transform. Defaults to 20000.
        filter_type (str, optional): Filter type ("lorenz" or "gaussian"). Defaults to "lorenz".
        nuclear_quantum_factor (float, optional): Correction factor for nuclear quantum effects. Defaults to 0.96.
        save_plots (Optional[List[str]], optional): List of 3 paths for iso, aniso, and low-freq plots. Defaults to None.
        save_data (Optional[str], optional): Path to save the raw array data. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - wavenumber: Array of evaluated frequencies (cm^-1).
            - total_iso: Isotropic Raman intensity.
            - total_aniso: Anisotropic Raman intensity.
            - low_range: Low-frequency Raman intensity.
    """
    weight = cutoff_z(h2o[..., 2], z0, zc, zw)

    corr_iso_intra, corr_aniso_intra, corr_iso_inter, corr_aniso_inter = calculate_corr_polar(
        atomic_polar, weight, h2o, cells, window, rc=rc
    )
    corr_iso_intra = np.sum(corr_iso_intra[:, : h2o.shape[1]], axis=1)
    corr_aniso_intra = np.sum(corr_aniso_intra[:, : h2o.shape[1]], axis=1)
    corr_iso_inter = np.sum(corr_iso_inter[:, : h2o.shape[1]], axis=1)
    corr_aniso_inter = np.sum(corr_aniso_inter[:, : h2o.shape[1]], axis=1)

    wavenumber = np.array(calculate_raman(corr_iso_intra, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type))[0]
    raman_iso_intra = np.array(calculate_raman(corr_iso_intra, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type))[1]
    raman_aniso_intra = np.array(calculate_raman(corr_aniso_intra, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type))[1]
    raman_iso_inter = np.array(calculate_raman(corr_iso_inter, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type))[1]
    raman_aniso_inter = np.array(calculate_raman(corr_aniso_inter, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type))[1]

    total_iso = raman_iso_intra + raman_iso_inter
    total_aniso = raman_aniso_intra + raman_aniso_inter
    low_range = wavenumber * (raman_aniso_intra + raman_aniso_inter) / 1000

    if save_plots is not None and len(save_plots) == 3:
        scaled_wavenumber = wavenumber * nuclear_quantum_factor

        # Isotropic Raman
        xlim_min, xlim_max = 2800, 4000
        norm_iso = _normalize_for_plot(scaled_wavenumber, total_iso, xlim_min, xlim_max)
        plt.plot(scaled_wavenumber, norm_iso, label="interfacial water", scalex=1.5, scaley=2.2)
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(0, 1)
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("Raman spectra (iso)")
        plt.savefig(save_plots[0], dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

        # Anisotropic Raman
        norm_aniso = _normalize_for_plot(scaled_wavenumber, total_aniso, xlim_min, xlim_max)
        plt.plot(scaled_wavenumber, norm_aniso, label="interfacial water", scalex=1.5, scaley=2.2)
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(0, 1)
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("Raman spectra (aniso)")
        plt.savefig(save_plots[1], dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

        # Low-frequency Raman
        xlim_min_low, xlim_max_low = 0, 2500
        norm_low = _normalize_for_plot(scaled_wavenumber, low_range, xlim_min_low, xlim_max_low)
        plt.plot(scaled_wavenumber, norm_low, label="interfacial water", scalex=1.5, scaley=2.2)
        plt.xlim(xlim_min_low, xlim_max_low)
        plt.ylim(0, 1)
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("Low-frequency Raman spectra (aniso)")
        plt.savefig(save_plots[2], dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()
        
    if save_data is not None:
        np.savetxt(save_data, np.array([wavenumber, total_iso, total_aniso, low_range]).T)

    return wavenumber, total_iso, total_aniso, low_range


def compute_surface_sfg_h2o(
    h2o: np.ndarray,
    cells: np.ndarray,
    atomic_dipole: np.ndarray,
    atomic_polar: np.ndarray,
    dt: float = 0.0005,
    window: int = 2000,
    z0: float = 25.0,
    zc: float = 7.5,
    zw: float = 0.5,
    rc: float = 6.0,
    width: int = 240,
    temperature: float = 330.0,
    M: int = 20000,
    filter_type: str = "lorenz",
    nuclear_quantum_factor: float = 0.96,
    save_plot: Optional[str] = None,
    save_data: Optional[str] = None,
):
    """
    Compute and optionally plot/save the surface SFG spectrum (xxz/yyz).

    Args:
        h2o (np.ndarray): Interfacial water molecule coordinates.
        cells (np.ndarray): Simulation box array.
        atomic_dipole (np.ndarray): Atomic dipole array.
        atomic_polar (np.ndarray): Atomic polarizability tensor array.
        dt (float, optional): Time step between frames in ps. Defaults to 0.0005.
        window (int, optional): Window size for correlation calculation. Defaults to 2000.
        z0 (float, optional): Z cutoff center to isolate surface water. Defaults to 25.0.
        zc (float, optional): Z cutoff width to isolate surface water. Defaults to 7.5.
        zw (float, optional): Z cutoff smoothing parameter. Defaults to 0.5.
        rc (float, optional): Cutoff radius for correlation function. Defaults to 6.0.
        width (int, optional): Smoothing width parameter for the SFG spectrum. Defaults to 240.
        temperature (float, optional): Simulation temperature in Kelvin. Defaults to 330.0.
        M (int, optional): Number of points for Fourier transform. Defaults to 20000.
        filter_type (str, optional): Filter type ("lorenz" or "gaussian"). Defaults to "lorenz".
        nuclear_quantum_factor (float, optional): Correction factor for nuclear quantum effects. Defaults to 0.96.
        save_plot (Optional[str], optional): Path to save the plot figure. Defaults to None.
        save_data (Optional[str], optional): Path to save the raw array data. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - wavenumber: Array of evaluated frequencies (cm^-1).
            - sfg_imag_h2o: The computed imaginary part of the SFG susceptibility (Im[chi^(2)]).
    """
    weight = cutoff_z(h2o[..., 2], z0, zc, zw)

    corr_intra, corr_inter = cal_corr_sfg_method2(
        atomic_polar[..., 0, 0] + atomic_polar[..., 1, 1], 
        atomic_dipole[..., 2], 
        weight, 
        h2o, 
        cells, 
        window, 
        rc=rc
    )
    corr_h2o = np.sum(corr_intra[:, :h2o.shape[1]], axis=1) + np.sum(corr_inter[:, :h2o.shape[1]], axis=1)

    wavenumber = np.array(calculate_sfg(corr_h2o, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type))[0]
    sfg_imag_h2o = np.array(calculate_sfg(corr_h2o, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type))[1]

    if save_plot is not None:
        scaled_wavenumber = wavenumber * nuclear_quantum_factor
        xlim_min, xlim_max = 2800, 4000
        norm_sfg = _normalize_for_plot(scaled_wavenumber, sfg_imag_h2o, xlim_min, xlim_max)
        
        plt.plot(scaled_wavenumber, norm_sfg, label="interfacial water", scalex=1.5, scaley=2.2)
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(-1, 1)
        plt.xlabel(r"Wavenumber($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Im[$\chi^{(2)}$ (a.u.)]", fontdict={"size": 12})
        plt.legend()
        plt.title("SFG spectra (xxz)")
        plt.savefig(save_plot, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

    if save_data is not None:
        np.savetxt(save_data, np.array([wavenumber, sfg_imag_h2o]).T)

    return wavenumber, sfg_imag_h2o


def set_cells_h2o(stc_list: List[Atoms], cell: List[float]):
    """
    Write the length and pbc conditions to the bounding box.

    Args:
        stc_list (List[Atoms]): A list of ASE Atoms objects representing simulation frames.
        cell (List[float]): Defining the dimensions/parameters of the simulation box.

    Returns:
        List[Atoms]: The modified list of ASE Atoms objects with updated cell/PBC info.
    """
    for stc in stc_list:
        stc.set_cell(cell)
        stc.set_pbc(True)
    return stc_list


def get_lumped_wacent_poses_rel_h2o(stc: Atoms, elem_symbol: str, cutoff: float = 1.0, expected_cn: int = 4):
    """
    Determine the positions of the Wannier centers around a target element (e.g., O) 
    and lump them into a mean relative displacement centroid.

    Args:
        stc (Atoms): ASE Atoms object for a specific simulation frame.
        elem_symbol (str): Chemical symbol of the element to analyze.
        cutoff (float, optional): Neighbor cutoff radius in angstroms. Defaults to 1.0.
        expected_cn (int, optional): Expected coordination number of Wannier centers. Defaults to 4.

    Returns:
        np.ndarray: Matrix of relative lumped Wannier centroid vectors for each targeted atom.
    """
    elem_idx = np.where(stc.symbols == elem_symbol)[0]
    wacent_idx = np.where(stc.symbols == "X")[0]
    elem_poses = stc.positions[elem_idx]
    wacent_poses = stc.positions[wacent_idx]

    cellpar = stc.cell.cellpar()
    assert cellpar is not None
    dist_mat = distance_array(elem_poses, wacent_poses, box=cellpar)

    lumped_wacent_poses_rel = []
    for elem_entry, dist_vec in enumerate(dist_mat):
        bool_vec = dist_vec < cutoff
        cn = np.sum(bool_vec)

        neig_wacent_poses = wacent_poses[bool_vec, :]
        neig_wacent_poses_rel = neig_wacent_poses - elem_poses[elem_entry]
        neig_wacent_poses_rel = minimize_vectors(neig_wacent_poses_rel, box=cellpar)
        lumped_wacent_pos_rel = neig_wacent_poses_rel.mean(axis=0)

        if cn != expected_cn:
            print(
                "The atom index is :",
                elem_idx[elem_entry],
                "The atom position is ",
                elem_poses[elem_entry],
                "The coordination number is :",
                cn,
            )
        lumped_wacent_poses_rel.append(lumped_wacent_pos_rel)
    lumped_wacent_poses_rel = np.stack(lumped_wacent_poses_rel)
    return lumped_wacent_poses_rel


def set_lumped_wfc_h2o(stc_list: List[Atoms], lumped_dict: Dict[str, int]):
    """
    Use get_lumped_wacent_poses_rel every frame to compute and return the absolute 
    coordinates of Wannier centroids.

    Args:
        stc_list (List[Atoms]): List of ASE Atoms objects for simulation frames.
        lumped_dict (Dict[str, int]): Dictionary where keys are elemental symbols and 
                                      values are their expected coordination numbers.

    Returns:
        np.ndarray: Evaluated coordinates of the lumped Wannier centroids across all frames.
    """
    X_pos = []
    for stc in stc_list:
        for elem_symbol, expected_cn in lumped_dict.items():
            lumped_wacent_poses_rel = get_lumped_wacent_poses_rel_h2o(
                stc=stc, elem_symbol=elem_symbol, cutoff=1.0, expected_cn=expected_cn
            )
            elem_pos = stc.get_positions()[stc.symbols == elem_symbol]
            X_pos.append(elem_pos + lumped_wacent_poses_rel)

    wfc_pos = np.reshape(X_pos, (len(stc_list), -1))
    return wfc_pos
