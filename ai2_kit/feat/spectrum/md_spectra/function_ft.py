from typing import Optional
import numpy as np

def apply_gussian_filter(corr: np.ndarray, width: float):
    """
    Apply gaussian filter. Parameter `width` means the smoothing width.
    """
    nmax = corr.shape[0] - 1
    return corr * np.exp(-.5 * (0.5 * width * np.arange(nmax + 1) / nmax)**2)

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

def _FFT_OE(C: np.ndarray, DTH: float, M: int):
    NMAX = C.shape[0] - 1
    NU = np.arange(M + 1)
    THETA = NU * DTH
    # Even coordinates
    CE = _range_fft(C[:-1:2], n = int(M / 2)).real # type: ignore
    CE = np.concatenate([CE, CE, CE[0:1]]) + C[NMAX] * np.cos(THETA * NMAX)
    # Odd coordinates
    CO = (_range_fft(C[1::2], n = int(M / 2)) * np.exp(-THETA[:int(M / 2)] * 1j)).real # type: ignore
    CO = np.concatenate([CO, -CO, CO[0:1]])
    return CE, CO

def FT(DT: float, C: np.ndarray, M: Optional[int] = None) -> np.ndarray:
    """
    The same as FILONC while `DOM = 2\pi / (M * DT)` (or `OMEGA_MAX = 2\pi / DT`).
    This is implemented by FFT.

    Parameters
    -----
    C: ndarray, the correlation function.
    DT: float, time interval between points in C.
    M: Optional[int], number of intervals on the frequency axis.
    `M = NMAX` by default.

    Return
    -----
    freq: float, frequency. `freq = 1 / (M * DT)` 
    CHAT: np.ndarray, the 1-d cosine transform.
    """
    NMAX = C.shape[0] - 1
    assert NMAX % 2 == 0, 'NMAX is not even!'
    if M is None:
        M = NMAX
    elif M % 2 != 0:
        M += 1
    DTH = 2 * np.pi / M
    NU = np.arange(M + 1)
    THETA = NU * DTH
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
    CE, CO = _FFT_OE(C, DTH, M)
    CE -= 0.5 * (C[0] + C[NMAX] * np.cos(THETA * NMAX))
    CHAT = 2.0 * (ALPHA * C[NMAX] * np.sin ( THETA * NMAX ) + BETA * CE + GAMMA * CO) * DT
    freq = 1 / (M * DT)
    return freq, CHAT

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

def calculate_ir(corr: np.ndarray, width: float, dt_ps: float, temperature: float, M: Optional[int] = None):
    nmax = corr.shape[0] - 1
    if nmax % 2 != 0:
        nmax -= 1
        corr = corr[:-1]
    tmax = nmax * dt_ps
    print('nmax      =', nmax)
    print('dt   (ps) =', dt_ps)
    print('tmax (ps) =', tmax)
    print("width     = ", width)
    width = width * tmax / 100.0 * 3
    C = apply_gussian_filter(corr, width)
    freq_ps, CHAT = FT(dt_ps, C, M)
    d_omega, CHAT = change_unit_ir(freq_ps, CHAT, temperature)
    return np.stack([np.arange(CHAT.shape[0]) * d_omega, CHAT], axis = 1)

def calculate_raman(corr: np.ndarray, width: float, dt_ps: float, temperature: float, M: Optional[int] = None):
    nmax = corr.shape[0] - 1
    if nmax % 2 != 0:
        nmax -= 1
        corr = corr[:-1]
    tmax = nmax * dt_ps        # ps
    print('nmax      =', nmax)
    print('dt   (ps) =', dt_ps)
    print('tmax (ps) =', tmax)
    print("width     = ", width)
    width = width * tmax / 100.0 * 3.0
    C = apply_gussian_filter(corr, width)
    freq_ps, CHAT = FT(dt_ps, C, M)
    d_omega, CHAT = change_unit_raman(freq_ps, CHAT, temperature)
    return np.stack([np.arange(CHAT.shape[0]) * d_omega, CHAT], axis = 1)

def calculate_sfg(corr: np.ndarray, width: int, dt_ps: float, temperature: float):
    nmax = corr.shape[0] - 1
    if nmax % 2 != 0:
        nmax -= 1
        corr = corr[:-1]
    tmax = nmax * dt_ps
    # dom = 2. * np.pi / tmax
    print('nmax  =', nmax)
    print('dt    =', dt_ps)
    print('tmax  =', tmax)
    print("width = ", width)
    width = width * tmax / 100.0 * 3.0
    C = apply_gussian_filter(corr, width)
    freq, CHAT = FT(dt_ps, C, 10000)
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
    CHAT *= -unit_all * freq * 1e4 * np.arange(CHAT.shape[0])
    d_omega = 1e10 * freq / cc
    return np.stack([np.arange(CHAT.shape[0]) * d_omega, CHAT], axis = 1)



