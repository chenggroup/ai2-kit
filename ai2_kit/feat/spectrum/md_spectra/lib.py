import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import dpdata
from MDAnalysis.lib.distances import distance_array, minimize_vectors

from .function_cal_corr import (
    calculate_corr_vdipole,
    calculate_corr_polar,
    cal_corr_sfg_method2,
    cutoff_z
)
from .function_ft import calculate_ir, calculate_raman, calculate_sfg
from .function_prepare import find_h2o, do_pbc, calculate_dipole


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


from ase import Atoms
from typing import Dict

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