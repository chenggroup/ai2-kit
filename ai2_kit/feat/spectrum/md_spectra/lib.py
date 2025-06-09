import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import dpdata
from MDAnalysis.lib.distances import distance_array, minimize_vectors

from .function_cal_corr import calculate_corr_vdipole, cal_range_dipole_polar, cal_weighted_dipole, cal_weighted_polar, cal_corr_sfg, calculate_corr_polar
from .function_ft import calculate_ir, calculate_raman, calculate_sfg
from .function_prepare import find_h2o, do_pbc, calculate_dipole, k_nearest


def compute_ir_spectrum_h2o(
    atomic_dipole: np.ndarray,
    dt: float = 0.0005,
    window: int = 50000,
    width: int = 25,
    temperature: float = 330.0,
    save_plot: Optional[str] = None,
    save_data: Optional[str] = None
):
    """
    Compute the IR spectrum from preprocessed atomic dipole data, and optionally save the plot and data.

    Parameters
    ----------
    atomic_dipole : np.ndarray
        Atomic dipole array of shape (n_steps, n_atoms, 3), already preprocessed.
    dt : float, optional
        Time step in picoseconds. Default is 0.0005.
    window : int, optional
        Window size for correlation calculation. Default is 50000.
    width : int, optional
        Width parameter for IR calculation. Default is 25.
    temperature : float, optional
        Temperature in Kelvin. Default is 330.0.
    save_plot : Optional[str], optional
        File path to save the IR plot. If None, the plot is not saved.
    save_data : Optional[str], optional
        File path to save the IR data as text. If None, the data is not saved.

    Returns
    -------
    ir : np.ndarray
        The computed IR spectrum, shape (n_points, 2).
    """
    corr_total = calculate_corr_vdipole(atomic_dipole, dt, window)
    ir = calculate_ir(corr_total, width=width, dt_ps=dt, temperature=temperature)

    plt.plot(ir[:, 0], ir[:, 1], label=r'$H_2O$', scalex=1.5, scaley=2.2)
    plt.xlim((0, 4000.))
    plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict={'size': 12})
    plt.ylabel(r'$n(\omega)\alpha(\omega) (10^3 cm^{-1})$', fontdict={'size': 12})
    plt.legend()
    plt.title("IR spectra")
    if save_plot is not None:
        plt.savefig(save_plot, dpi=300, facecolor='white', bbox_inches='tight')
    if save_data is not None:
        np.savetxt(save_data, ir)
    return ir


def compute_raman_spectra_h2o(
    atomic_polar: np.ndarray,
    dt: float = 0.0005,
    window: int = 50000,
    width_iso: int = 25,
    width_aniso: int = 240,
    temperature: float = 330.0,
    save_plots: Optional[List[str]] = None,
    save_datas: Optional[List[str]] = None
):
    """
    Compute isotropic and anisotropic Raman spectra from preprocessed atomic polarizability data,
    and optionally save plots and data to files.

    Parameters
    ----------
    atomic_polar : np.ndarray
        Preprocessed atomic polarizability array of shape (n_steps, n_molecules, 3, 3).
    dt : float, optional
        Time step in picoseconds. Default is 0.0005.
    window : int, optional
        Window size for correlation calculation. Default is 50000.
    width_iso : int, optional
        Width parameter for isotropic Raman calculation. Default is 25.
    width_aniso : int, optional
        Width parameter for anisotropic Raman calculation. Default is 240.
    temperature : float, optional
        Temperature in Kelvin. Default is 330.0.
    save_plots : Optional[List[str]], optional
        List of file paths to save the plots: [iso_path, aniso_path, aniso_low_path].
        If None, plots are not saved.
    save_datas : Optional[List[str]], optional
        List of file paths to save the data: [iso_path, aniso_path, aniso_low_path].
        If None, data is not saved.

    Returns
    -------
    Tuple of (raman_iso, raman_aniso, raman_aniso_low) spectra as numpy arrays.
    """
    corr_total = calculate_corr_polar(atomic_polar, window)
    raman_iso = calculate_raman(corr_total[0], width=width_iso, dt_ps=dt, temperature=temperature)
    raman_aniso = calculate_raman(corr_total[1], width=width_aniso, dt_ps=dt, temperature=temperature)
    raman_aniso_low = np.column_stack((raman_aniso[:, 0], raman_aniso[:, 1] * raman_aniso[:, 0] / 1000))

    # Plot and save isotropic Raman
    plt.plot(raman_iso[:, 0], raman_iso[:, 1], label=r'$H_2O$, iso', scalex=1.5, scaley=2.2)
    plt.xlim((2800, 4000.))
    plt.ylim((0, 1))
    plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict={'size': 12})
    plt.ylabel(r'Intensity', fontdict={'size': 12})
    plt.legend()
    plt.title("Raman spectra (iso)")
    if save_plots is not None and len(save_plots) > 0 and save_plots[0]:
        plt.savefig(save_plots[0], dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    if save_datas is not None and len(save_datas) > 0 and save_datas[0]:
        np.savetxt(save_datas[0], raman_iso)

    # Plot and save anisotropic Raman
    plt.plot(raman_aniso[:, 0], raman_aniso[:, 1], label=r'$H_2O$, aniso', scalex=1.5, scaley=2.2)
    plt.xlim((2800, 4000.))
    plt.ylim((0, 3))
    plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict={'size': 12})
    plt.ylabel(r'Intensity', fontdict={'size': 12})
    plt.legend()
    plt.title("Raman spectra (aniso)")
    if save_plots is not None and len(save_plots) > 1 and save_plots[1]:
        plt.savefig(save_plots[1], dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    if save_datas is not None and len(save_datas) > 1 and save_datas[1]:
        np.savetxt(save_datas[1], raman_aniso)

    # Plot and save low-frequency anisotropic Raman
    plt.plot(raman_aniso_low[:, 0], raman_aniso_low[:, 1], label=r'$H_2O$, aniso_low', scalex=1.5, scaley=2.2)
    plt.xlim((0, 2500.))
    plt.ylim((0, 8))
    plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict={'size': 12})
    plt.ylabel(r'intensity', fontdict={'size': 12})
    plt.legend()
    plt.title("Low-frequency Raman spectra (aniso)")
    if save_plots is not None and len(save_plots) > 2 and save_plots[2]:
        plt.savefig(save_plots[2], dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    if save_datas is not None and len(save_datas) > 2 and save_datas[2]:
        np.savetxt(save_datas[2], raman_aniso_low)

    return raman_iso, raman_aniso, raman_aniso_low


def compute_atomic_dipole_h2o(
    traj: "dpdata.System",
    wannier: np.ndarray,
    type_O: int = 1,
    type_H: int = 2,
    r_bond: float = 1.3,
    a0: float = 0.52917721067,
    save_datas: Optional[str] = None,
):
    """
    Compute atomic dipole moments for water molecules from trajectory and Wannier center data,
    and optionally save the computed h2o coordinates and atomic dipole array to files.

    Parameters
    ----------
    traj : dpdata.System
        The trajectory object loaded by dpdata, containing "coords", "cells", and "atom_types".
    wannier : np.ndarray
        Wannier center coordinates, shape (n_frames, n_wannier, 3).
    type_O : int, optional
        Atomic type index for oxygen. Default is 1.
    type_H : int, optional
        Atomic type index for hydrogen. Default is 2.
    r_bond : float, optional
        O-H bond cutoff distance for water identification. Default is 1.3.
    a0 : float, optional
        Bohr radius in angstroms for unit conversion. Default is 0.52917721067.
    save_datas : Optional[List[str]], optional
        List of file paths to save the data: [h2o, atomic_dipole].
        If None, data is not saved.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        h2o: The computed water molecule center of mass coordinates, shape (n_frames, n_water, 3).
        atomic_dipole: The computed atomic dipole array, shape (n_frames, n_water, 3).
    """
    coords = traj["coords"]
    cells = traj["cells"]
    types = traj["atom_types"]
    coords = do_pbc(coords, cells)
    coords_O = coords[:, types == type_O, :]
    coords_H = coords[:, types == type_H, :]

    h2o_mask = find_h2o(coords_O[0], coords_H[0], cells[0], r_bond=r_bond)
    idx_h2o_H = k_nearest(coords_O[0, h2o_mask, :], coords_H[0], cells[0], 2).flatten()
    h2o_O = coords_O[:, h2o_mask, :]
    h2o_H = coords_H[:, idx_h2o_H, :].reshape(traj.get_nframes(), -1, 2, 3)
    h2o = (h2o_O * 16 + np.sum(h2o_H, axis=2)) / 18

    if save_datas is not None and len(save_datas) > 0 and save_datas[0]:
        np.save(save_datas[0], h2o)

    # Assume wannier is already reshaped to (n_frames, n_wannier, 3)
    wannier_sel = wannier[:, h2o_mask, :]
    atomic_dipole = calculate_dipole(h2o_O, coords_H, cells, wannier_sel, r_bond) * np.sqrt(a0)

    if save_datas is not None and len(save_datas) > 1 and save_datas[1]:
        np.save(save_datas[1], atomic_dipole.reshape(traj.get_nframes(), -1))

    return h2o, atomic_dipole


def extract_atomic_polar_from_traj_h2o(
    traj: dpdata.System,
    polar: np.ndarray,
    type_O: int = 1,
    type_H: int = 2,
    r_bond: float = 1.3,
    save_data: Optional[str] = None,
):
    """
    Extract atomic polarizability tensors for water molecules from trajectory and raw polarizability data.

    Parameters
    ----------
    traj : dpdata.System
        The trajectory object loaded by dpdata, containing "coords", "cells", and "atom_types".
    polar : np.ndarray
        Raw polarizability tensor array, shape (n_frames, n_polar, 3, 3) or (n_frames * n_polar, 3, 3) before reshape.
    type_O : int, optional
        Atomic type index for oxygen. Default is 1.
    type_H : int, optional
        Atomic type index for hydrogen. Default is 2.
    r_bond : float, optional
        O-H bond cutoff distance for water identification. Default is 1.3.
    save_data : str, optional
        File path to save the atomic polarizability array, including the filename (e.g., "atomic_polar_wan.npy").
    Returns
    -------
    atomic_polar : np.ndarray
        The extracted atomic polarizability array, shape (n_frames, n_water, 3, 3).
    """
    coords = traj["coords"]
    cells = traj["cells"]
    types = traj["atom_types"]

    coords = do_pbc(coords, cells)
    coords_O = coords[:, types == type_O, :]
    coords_H = coords[:, types == type_H, :]
    h2o_mask = find_h2o(coords_O[0], coords_H[0], cells[0], r_bond=r_bond)

    if polar.ndim == 4:
        atomic_polar = -polar[:, h2o_mask, :, :]
    else:
        n_frames = coords.shape[0]
        atomic_polar = -polar.reshape(n_frames, -1, 3, 3)[:, h2o_mask, :, :]

    if save_data is not None:
        np.save(save_data, atomic_polar)

    return atomic_polar


def compute_surface_ir_spectra_h2o(
    h2o: np.ndarray,
    atomic_dipole: np.ndarray,
    dt: float = 0.0005,
    window: int = 50000,
    z1_min: float = 16.0,
    z1_max: float = 17.4,
    z2_min: float = 20.0,
    z2_max: float = 25.0,
    z3_min: float = 27.6,
    z3_max: float = 29.0,
    z_total_min: float = 16.0,
    z_total_max: float = 29.0,
    z_bin: float = 0.4,
    width: int = 25,
    temperature: float = 330.0,
    save_plot: Optional[str] = None,
    save_data: Optional[str] = None
):
    """
    Compute and optionally plot/save surface and bulk IR spectra for different z-ranges.

    Parameters
    ----------
    h2o : np.ndarray
        Water molecule positions, shape (n_frames, n_molecules, 3).
    atomic_dipole : np.ndarray
        Atomic dipole array, shape (n_frames, n_molecules, 3).
    dt : float, optional
        Time step in picoseconds. Default is 0.0005.
    window : int, optional
        Window size for correlation calculation. Default is 50000.
    z1_min, z1_max : float, optional
        z-range for surface region 1. Default is 16.0, 17.4.
    z2_min, z2_max : float, optional
        z-range for bulk region. Default is 20.0, 25.0.
    z3_min, z3_max : float, optional
        z-range for surface region 3. Default is 27.6, 29.0.
    z_total_min, z_total_max : float, optional
        z-range for total region. Default is 16.0, 29.0.
    z_bin : float, optional
        Bin width for z-range selection. Default is 0.4.
    width : int, optional
        Width parameter for IR calculation. Default is 25.
    temperature : float, optional
        Temperature in Kelvin. Default is 330.0.
    save_plot : str, optional
        File path to save the plot, including the filename (e.g., "ir_sp.png").
    save_data : str, optional
        File path to save the IR spectra data, including the filename (e.g., "ir_sp.dat").

    Returns
    -------
    Tuple of IR spectra arrays for surface and bulk regions.
    """
    # Select z-coordinates
    z_coords = h2o[..., [2]]

    # Dipole selection by z-range
    # total_dipole = cal_range_dipole_polar(z_coords, atomic_dipole, z_total_min, z_total_max, z_bin)
    range_dipole1 = cal_range_dipole_polar(z_coords, atomic_dipole, z1_min, z1_max, z_bin)
    range_dipole2 = cal_range_dipole_polar(z_coords, atomic_dipole, z2_min, z2_max, z_bin)
    range_dipole3 = cal_range_dipole_polar(z_coords, atomic_dipole, z3_min, z3_max, z_bin)

    # Correlation and IR for xy
    corr_range1_xy = calculate_corr_vdipole(range_dipole1[:, [0, 1]], dt, window)
    corr_range2_xy = calculate_corr_vdipole(range_dipole2[:, [0, 1]], dt, window)
    corr_range3_xy = calculate_corr_vdipole(range_dipole3[:, [0, 1]], dt, window)

    ir_range1_xy = calculate_ir(corr_range1_xy, width=width, dt_ps=dt, temperature=temperature)
    ir_range2_xy = calculate_ir(corr_range2_xy, width=width, dt_ps=dt, temperature=temperature)
    ir_range3_xy = calculate_ir(corr_range3_xy, width=width, dt_ps=dt, temperature=temperature)

    # Correlation and IR for z
    corr_range1_z = calculate_corr_vdipole(range_dipole1[:, [2]], dt, window)
    corr_range2_z = calculate_corr_vdipole(range_dipole2[:, [2]], dt, window)
    corr_range3_z = calculate_corr_vdipole(range_dipole3[:, [2]], dt, window)

    ir_range1_z = calculate_ir(corr_range1_z, width=width, dt_ps=dt, temperature=temperature)
    ir_range2_z = calculate_ir(corr_range2_z, width=width, dt_ps=dt, temperature=temperature)
    ir_range3_z = calculate_ir(corr_range3_z, width=width, dt_ps=dt, temperature=temperature)

    # Prepare data for output and plotting
    s_surface_xy = (ir_range1_xy[:, 1] + ir_range3_xy[:, 1]) / 2
    s_surface_z = (ir_range1_z[:, 1] + ir_range3_z[:, 1]) / 2

    if save_plot is not None:
        plt.plot(ir_range1_xy[:, 0], s_surface_xy, label=r'surface $H_2O$ with s-polarized', scalex=1.5, scaley=2.2)
        plt.plot(ir_range2_xy[:, 0], ir_range2_xy[:, 1], label=r'bulk $H_2O$ with s-polarized', scalex=1.5, scaley=2.2)
        plt.plot(ir_range1_z[:, 0], s_surface_z, label=r'surface $H_2O$ with p-polarized', scalex=1.5, scaley=2.2)
        plt.plot(ir_range2_z[:, 0], ir_range2_z[:, 1], label=r'bulk $H_2O$ with p-polarized', scalex=1.5, scaley=2.2)
        plt.xlim((0, 4000.))
        plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict={'size': 12})
        plt.ylabel(r'$n(\omega)\alpha(\omega) (10^3 cm^{-1})$', fontdict={'size': 12})
        plt.legend()
        plt.title("IR spectra")
        plt.savefig(save_plot, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()

    if save_data is not None:
        # Save columns: wavenumber, surface_xy, bulk_xy, surface_z, bulk_z
        np.savetxt(
            save_data,
            np.column_stack([
                ir_range1_xy[:, 0],
                s_surface_xy,
                ir_range2_xy[:, 1],
                s_surface_z,
                ir_range2_z[:, 1]
            ])
        )

    return (ir_range1_xy, ir_range2_xy, ir_range3_xy, ir_range1_z, ir_range2_z, ir_range3_z)


def compute_surface_raman_h2o(
    h2o: np.ndarray,
    atomic_polar: np.ndarray,
    dt: float = 0.0005,
    window: int = 50000,
    z_total_min: float = 16.0,
    z_total_max: float = 29.0,
    z1_min: float = 16.0,
    z1_max: float = 17.4,
    z2_min: float = 20.0,
    z2_max: float = 25.0,
    z3_min: float = 27.6,
    z3_max: float = 29.0,
    z_bin: float = 0.4,
    width: int = 25,
    temperature: float = 330.0,
    save_plots: Optional[List[str]] = None,
    save_datas: Optional[List[str]] = None,
):
    """
    Compute and optionally plot/save surface and bulk Raman spectra for different z regions.

    Parameters
    ----------
    h2o : np.ndarray
        Water molecule coordinates, shape (n_frames, n_molecules, 3).
    atomic_polar : np.ndarray
        Atomic polarizability tensors, shape (n_frames, n_molecules, 3, 3).
    dt : float, optional
        Time step in picoseconds. Default is 0.0005.
    window : int, optional
        Window size for correlation calculation. Default is 50000.
    z_total_min, z_total_max, z1_min, z1_max, z2_min, z2_max, z3_min, z3_max : float, optional
        Z region boundaries for surface and bulk selection.
    z_bin : float, optional
        Bin width for z axis. Default is 0.4.
    width : int, optional
        Width parameter for Raman calculation. Default is 25.
    temperature : float, optional
        Temperature in Kelvin. Default is 330.0.
    save_plots : Optional[List[str]], optional
        List of file paths to save the plots: [iso, aniso, aniso_low]. If None, plots are not saved.
    save_datas : Optional[List[str]], optional
        List of file paths to save the data: [iso, aniso, aniso_low]. If None, data is not saved.
    show_plot : bool, optional
        Whether to display plots interactively. Default is False.

    Returns
    -------
    Tuple of computed spectra arrays for iso, aniso, and low-frequency aniso.
    """

    # Z axis is the 3rd column (index 2)
    z_axis = h2o[..., [2], None]
    total_polar = cal_range_dipole_polar(z_axis, atomic_polar, z_total_min, z_total_max, z_bin)
    range_polar1 = cal_range_dipole_polar(z_axis, atomic_polar, z1_min, z1_max, z_bin)
    range_polar2 = cal_range_dipole_polar(z_axis, atomic_polar, z2_min, z2_max, z_bin)
    range_polar3 = cal_range_dipole_polar(z_axis, atomic_polar, z3_min, z3_max, z_bin)

    corr_atomic = calculate_corr_polar(atomic_polar[:, 0], window)
    corr_total = calculate_corr_polar(total_polar, window)
    corr_range1 = calculate_corr_polar(range_polar1, window)
    corr_range2 = calculate_corr_polar(range_polar2, window)
    corr_range3 = calculate_corr_polar(range_polar3, window)

    # Isotropic
    raman_atomic_iso = calculate_raman(corr_atomic[0], width=width, dt_ps=dt, temperature=temperature)
    raman_total_iso = calculate_raman(corr_total[0], width=width, dt_ps=dt, temperature=temperature)
    raman_range1_iso = calculate_raman(corr_range1[0], width=width, dt_ps=dt, temperature=temperature)
    raman_range2_iso = calculate_raman(corr_range2[0], width=width, dt_ps=dt, temperature=temperature)
    raman_range3_iso = calculate_raman(corr_range3[0], width=width, dt_ps=dt, temperature=temperature)

    # Normalization
    SMAX = np.max(raman_atomic_iso[:8000, 1])
    SMAX = max(SMAX, np.max(raman_total_iso[:8000, 1]))
    SMAX = max(SMAX, np.max(raman_range1_iso[:8000, 1]))
    SMAX = max(SMAX, np.max(raman_range2_iso[:8000, 1]))
    SMAX = max(SMAX, np.max(raman_range3_iso[:8000, 1]))
    SMAX /= 10

    # Plot/save iso
    plt.plot(raman_total_iso[:, 0], raman_total_iso[:, 1] / SMAX, label=r'$H_2O$, total', scalex=1.5, scaley=2.2)
    plt.plot(raman_range1_iso[:, 0], (raman_range1_iso[:, 1] + raman_range3_iso[:, 1]) / 2 / SMAX, label=r'surface $H_2O$', scalex=1.5, scaley=2.2)
    plt.plot(raman_range2_iso[:, 0], raman_range2_iso[:, 1] / SMAX, label=r'bulk $H_2O$', scalex=1.5, scaley=2.2)
    plt.xlim((2800, 4000.))
    plt.ylim((0, 1))
    plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict={'size': 12})
    plt.ylabel(r'Intensity', fontdict={'size': 12})
    plt.legend()
    plt.title("Raman spectra (iso)")
    if save_plots is not None and len(save_plots) > 0 and save_plots[0]:
        plt.savefig(save_plots[0], dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    if save_datas is not None and len(save_datas) > 0 and save_datas[0]:
        np.savetxt(
            save_datas[0],
            np.column_stack((
                raman_total_iso[:, 0],
                raman_total_iso[:, 1] / SMAX,
                (raman_range1_iso[:, 1] + raman_range3_iso[:, 1]) / 2 / SMAX,
                raman_range2_iso[:, 1] / SMAX
            ))
        )

    # Anisotropic
    raman_atomic_aniso = calculate_raman(corr_atomic[1], width=width, dt_ps=dt, temperature=temperature)
    raman_total_aniso = calculate_raman(corr_total[1], width=width, dt_ps=dt, temperature=temperature)
    raman_range1_aniso = calculate_raman(corr_range1[1], width=width, dt_ps=dt, temperature=temperature)
    raman_range2_aniso = calculate_raman(corr_range2[1], width=width, dt_ps=dt, temperature=temperature)
    raman_range3_aniso = calculate_raman(corr_range3[1], width=width, dt_ps=dt, temperature=temperature)

    plt.plot(raman_total_aniso[:, 0], raman_total_aniso[:, 1] / SMAX, label=r'$H_2O$, total', scalex=1.5, scaley=2.2)
    plt.plot(raman_range1_aniso[:, 0], (raman_range1_aniso[:, 1] + raman_range3_aniso[:, 1]) / 2 / SMAX, label=r'surface $H_2O$', scalex=1.5, scaley=2.2)
    plt.plot(raman_range2_aniso[:, 0], raman_range2_aniso[:, 1] / SMAX, label=r'bulk $H_2O$', scalex=1.5, scaley=2.2)
    plt.xlim((2800, 4000.))
    plt.ylim((0, 3))
    plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict={'size': 12})
    plt.ylabel(r'Intensity', fontdict={'size': 12})
    plt.legend()
    plt.title("Raman spectra (aniso)")
    if save_plots is not None and len(save_plots) > 1 and save_plots[1]:
        plt.savefig(save_plots[1], dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    if save_datas is not None and len(save_datas) > 1 and save_datas[1]:
        np.savetxt(
            save_datas[1],
            np.column_stack((
                raman_total_aniso[:, 0],
                raman_total_aniso[:, 1] / SMAX,
                (raman_range1_aniso[:, 1] + raman_range3_aniso[:, 1]) / 2 / SMAX,
                raman_range2_aniso[:, 1] / SMAX
            ))
        )

    # Low-frequency
    low_total = raman_total_aniso[:, 1] * raman_total_aniso[:, 0] / 1000
    low_range1 = raman_range1_aniso[:, 1] * raman_range1_aniso[:, 0] / 1000
    low_range2 = raman_range2_aniso[:, 1] * raman_range2_aniso[:, 0] / 1000
    low_range3 = raman_range3_aniso[:, 1] * raman_range3_aniso[:, 0] / 1000

    plt.plot(raman_total_aniso[:, 0], low_total / SMAX, label=r'$H_2O$, total', scalex=1.5, scaley=2.2)
    plt.plot(raman_range1_aniso[:, 0], (low_range1 + low_range3) / 2 / SMAX, label=r'surface $H_2O$', scalex=1.5, scaley=2.2)
    plt.plot(raman_range2_aniso[:, 0], low_range2 / SMAX, label=r'bulk $H_2O$', scalex=1.5, scaley=2.2)
    plt.xlim((0, 2500.))
    plt.ylim((0, 8))
    plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict={'size': 12})
    plt.ylabel(r'intensity', fontdict={'size': 12})
    plt.legend()
    plt.title("Low-frequency Raman spectra (aniso)")
    if save_plots is not None and len(save_plots) > 2 and save_plots[2]:
        plt.savefig(save_plots[2], dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    if save_datas is not None and len(save_datas) > 2 and save_datas[2]:
        np.savetxt(
            save_datas[2],
            np.column_stack((
                raman_total_aniso[:, 0],
                raman_total_aniso[:, 1] / SMAX,
                (raman_range1_aniso[:, 1] + raman_range3_aniso[:, 1]) / 2 / SMAX,
                raman_range2_aniso[:, 1] / SMAX
            ))
        )

    return (
        (raman_total_iso, raman_range1_iso, raman_range2_iso, raman_range3_iso),
        (raman_total_aniso, raman_range1_aniso, raman_range2_aniso, raman_range3_aniso),
        (low_total, low_range1, low_range2, low_range3)
    )


def compute_surface_sfg_h2o(
    h2o: np.ndarray,
    atomic_dipole: np.ndarray,
    atomic_polar: np.ndarray,
    dt: float = 0.0005,
    window: int = 50000,
    z0: float = 22.5,
    zc: float = 2.5,
    zw: float = 2.6,
    rc: float = 6.75,
    width: int = 50,
    temperature: float = 330.0,
    save_plot: Optional[str] = None,
    save_data: Optional[str] = None
):
    """
    Compute and optionally plot/save the surface SFG (sum-frequency generation) spectrum.

    Parameters
    ----------
    h2o : np.ndarray
        Water molecule positions, shape (n_frames, n_molecules, 3).
    atomic_dipole : np.ndarray
        Atomic dipole array, shape (n_frames, n_molecules, 3).
    atomic_polar : np.ndarray
        Atomic polarizability array, shape (n_frames, n_molecules, 3, 3).
    dt : float, optional
        Time step in picoseconds. Default is 0.0005.
    window : int, optional
        Window size for correlation calculation. Default is 50000.
    z0 : float, optional
        Center position for the weighting function along the z-axis. Default is 22.5.
    zc : float, optional
        Characteristic width for the weighting function. Default is 2.5.
    zw : float, optional
        Width parameter for the weighting function. Default is 2.6.
    rc : float, optional
        Cutoff radius for the correlation calculation. Default is 6.75.
    width : int, optional
        Width parameter for SFG calculation. Default is 50.
    temperature : float, optional
        Temperature in Kelvin. Default is 330.0.
    save_plot : str, optional
        File path to save the plot, including the filename (e.g., "sfg.png").
    save_data : str, optional
        File path to save the SFG spectrum data, including the filename (e.g., "SFG.dat").

    Returns
    -------
    sfg : np.ndarray
        The computed SFG spectrum, shape (n_points, 2).
    """
    weighted_dipole = cal_weighted_dipole(h2o[..., [2]], atomic_dipole, z0, zc, zw)
    weighted_polar = cal_weighted_polar(h2o[..., [2], None], atomic_polar, z0, zc, zw)

    corr = cal_corr_sfg(
        weighted_polar[..., 0, 0] + weighted_polar[..., 1, 1],
        weighted_dipole[..., 2],
        h2o,
        window,
        rc=rc
    )
    sfg = calculate_sfg(corr, width=width, dt_ps=dt, temperature=temperature)

    if save_plot is not None:
        plt.plot(sfg[:, 0], sfg[:, 1], label=r'$H_2O$', scalex=1.5, scaley=2.2)
        plt.xlim((0, 4000.))
        plt.ylim((-0.12, 0.12))
        plt.xlabel(r'Wavenumber($\rm cm^{-1}$)', fontdict={'size': 12})
        plt.ylabel(r'Im[$\chi^{(2)}$]', fontdict={'size': 12})
        plt.legend()
        plt.title("SFG in xxz and yyz")
        plt.savefig(save_plot, dpi=100, facecolor='white', bbox_inches='tight')
        plt.close()

    if save_data is not None:
        np.savetxt(save_data, sfg)

    return sfg


from ase import Atoms
from typing import List, Dict

def set_cells_h2o(stc_list: List[Atoms], cell: List[float]):
    """
    write the length and pbc conditions of box

    Parameters
    ----------
    stc_list : List[Atoms]
        List of ASE Atoms objects representing trajectory frames.
    cell : List[float]
        Simulation cell parameters (e.g., [a, b, c]).

    Returns
    -------
    List[Atoms]
        The modified list with updated cell and PBC.
    """
    for stc in stc_list:
        stc.set_cell(cell)
        stc.set_pbc(True)
    return stc_list


def get_lumped_wacent_poses_rel_h2o(
    stc: Atoms,
    elem_symbol: str,
    cutoff: float = 1.0,
    expected_cn: int = 4
):
    """
    determine the positions of the wannaier centers around O and sum it into the wannaier centroid

    Parameters
    ----------
    stc : Atoms
        ASE Atoms object for a single frame.
    elem_symbol : str
        The element symbol (e.g., 'O') whose neighboring Wannier centers are to be lumped.
    cutoff : float, optional
        Distance cutoff for selecting neighboring Wannier centers. Default is 1.0.
    expected_cn : int, optional
        Expected coordination number (number of Wannier centers per atom). Default is 4.

    Returns
    -------
    np.ndarray
        Array of mean relative positions for each target atom.
    """
    elem_idx   = np.where(stc.symbols == elem_symbol)[0]
    wacent_idx = np.where(stc.symbols == 'X')[0]
    elem_poses = stc.positions[elem_idx]
    wacent_poses = stc.positions[wacent_idx]
    
    cellpar = stc.cell.cellpar()
    assert cellpar is not None
    #dist_matrix
    dist_mat = distance_array(elem_poses, wacent_poses, box=cellpar)

    #each row get distance and select the candidates
    lumped_wacent_poses_rel = []
    for elem_entry, dist_vec in enumerate(dist_mat):
        #print(_elem_idx)
        bool_vec = (dist_vec < cutoff)
        cn = np.sum(bool_vec)
        
        # modify neighbor wannier centers coords relative to the center element atom
        neig_wacent_poses = wacent_poses[bool_vec, :]
        neig_wacent_poses_rel = neig_wacent_poses - elem_poses[elem_entry]
        neig_wacent_poses_rel = minimize_vectors(neig_wacent_poses_rel, box=cellpar)
        lumped_wacent_pos_rel = neig_wacent_poses_rel.mean(axis=0)

        if cn != expected_cn:
            print("The atom index is :", elem_idx[elem_entry], "The atom position is ", elem_poses[elem_entry],"The coordination number is :", cn)
        lumped_wacent_poses_rel.append(lumped_wacent_pos_rel)
    lumped_wacent_poses_rel = np.stack(lumped_wacent_poses_rel)
    return lumped_wacent_poses_rel


def set_lumped_wfc_h2o(stc_list: List[Atoms], lumped_dict: Dict[str, int]):
    """
    use get_lumped_wacent_poses_rel function every frame and return the coordinates file of wannier centroids in npy form

    Parameters
    ----------
    stc_list : List[Atoms]
        List of ASE Atoms objects representing trajectory frames.
    lumped_dict : Dict[str, int]
        Dictionary mapping element symbols (e.g., 'O') to expected Wannier center counts per atom.

    Returns
    -------
    np.ndarray
        Array of Wannier centroid positions for all frames, shape (n_frames, n_atoms * 3).
    """
    X_pos = []
    for stc in stc_list:
        for elem_symbol, expected_cn in lumped_dict.items():
            lumped_wacent_poses_rel = get_lumped_wacent_poses_rel_h2o(stc=stc, elem_symbol=elem_symbol, cutoff = 1.0, expected_cn=expected_cn)
            elem_pos = stc.get_positions()[stc.symbols==elem_symbol]
            X_pos.append(elem_pos + lumped_wacent_poses_rel)

    wfc_pos = np.reshape(X_pos, (len(stc_list), -1))
    return wfc_pos