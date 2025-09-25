import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
import dpdata
from MDAnalysis.lib.distances import distance_array, minimize_vectors

from .function_cal_corr import (
    calculate_corr_vdipole,
    cal_range_dipole_polar,
    cal_weighted_dipole,
    cal_weighted_polar,
    cal_corr_sfg,
)
from .function_ft import calculate_ir, calculate_sfg
from .function_prepare import find_h2o, do_pbc, calculate_dipole, k_nearest
from .function_atomic import (
    calculate_corr_vdipole_atomic,
    cutoff_z,
    calculate_ir_atomic,
    calculate_corr_polar_atomic,
    calculate_raman_atomic,
)


def compute_bulk_ir_h2o(
    h2o: np.ndarray,
    cells: np.ndarray,
    atomic_dipole: np.ndarray,
    dt: float = 0.0005,
    window: int = 2000,
    z0: float = 21.5,
    zc: float = 5.0,
    zw: float = 0.5,
    rc: float = 6.0,
    width: int = 240,
    temperature: float = 330.0,
    M: int = 20000,
    filter_type: str = "lorenz",
    save_plot: Optional[str] = None,
    save_data: Optional[str] = None,
):
    """
    Compute and optionally save the bulk IR spectrum from atomic dipole and geometry data.

    Parameters
    ----------
    h2o : np.ndarray
        Water molecule coordinates, shape (n_frames, n_molecules, 3).
    cells : np.ndarray
        Simulation box array, shape (n_frames, 3, 3).
    atomic_dipole : np.ndarray
        Atomic dipole array, shape (n_frames, n_molecules, 3).
    dt : float, optional
        Time step in picoseconds. Default is 0.0005.
    window : int, optional
        Window size for correlation calculation. Default is 2000.
    z0 : float, optional
        Z cutoff center for weight calculation. Default is 21.5.
    zc : float, optional
        Z cutoff width for weight calculation. Default is 5.0.
    zw : float, optional
        Z cutoff smoothing parameter. Default is 0.5.
    rc : float, optional
        Cutoff radius for correlation calculation. Default is 6.0.
    width : int, optional
        Width parameter for IR calculation. Default is 240.
    temperature : float, optional
        Temperature in Kelvin. Default is 330.0.
    M : int, optional
        Number of points for IR calculation. Default is 20000.
    filter_type : str, optional
        Filter type for IR calculation. Default is "lorenz".
    save_plot : Optional[str], optional
        File path to save the plot. If None, the plot is not saved.
    save_data : Optional[str], optional
        File path to save the data. If None, the data is not saved.

    Returns
    -------
    Tuple of (wavenumber, ir) as np.ndarray.
    """

    weight = cutoff_z(h2o[..., 2], z0, zc, zw)
    weight = np.ones((h2o.shape[0], h2o.shape[1])) - weight

    corr_intra, corr_inter = calculate_corr_vdipole_atomic(atomic_dipole, weight, h2o, cells, dt, window, rc=rc)
    corr_intra = np.sum(corr_intra, axis=1)
    corr_inter = np.sum(corr_inter, axis=1)

    wavenumber = np.array(
        calculate_ir_atomic(
            np.sum(corr_intra, axis=1), width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type
        )
    )[0]
    ir = np.array(
        calculate_ir_atomic(
            np.sum(corr_intra, axis=1) + np.sum(corr_inter, axis=1),
            width=width,
            dt_ps=dt,
            temperature=temperature,
            M=M,
            filter_type=filter_type,
        )
    )[1]

    plt.plot(wavenumber, ir, label=r"$H_2O$", scalex=1.5, scaley=2.2)
    plt.xlim((0, 4000.0))
    plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
    plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
    plt.legend()
    plt.title("IR spectra")
    if save_plot is not None:
        plt.savefig(save_plot, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close()
    if save_data is not None:
        np.save(save_data, np.array([wavenumber, ir]))

    return wavenumber, ir


def compute_bulk_raman_h2o(
    h2o: np.ndarray,
    cells: np.ndarray,
    atomic_polar: np.ndarray,
    dt: float = 0.0005,
    window: int = 2000,
    z0: float = 21.5,
    zc: float = 5.0,
    zw: float = 0.5,
    rc: float = 6.0,
    width: int = 240,
    temperature: float = 330.0,
    M: int = 20000,
    filter_type: str = "lorenz",
    save_plots: Optional[List[str]] = None,
    save_data: Optional[str] = None,
):
    """
    Compute and optionally save the bulk Raman spectra (isotropic, anisotropic, and low-frequency) from atomic polarizability and geometry data.

    Parameters
    ----------
    h2o : np.ndarray
        Water molecule coordinates, shape (n_frames, n_molecules, 3).
    cells : np.ndarray
        Simulation box array, shape (n_frames, 3, 3).
    atomic_polar : np.ndarray
        Atomic polarizability tensor, shape (n_frames, n_molecules, 3, 3).
    dt : float, optional
        Time step in picoseconds. Default is 0.0005.
    window : int, optional
        Window size for correlation calculation. Default is 2000.
    z0 : float, optional
        Z cutoff center for weight calculation. Default is 21.5.
    zc : float, optional
        Z cutoff width for weight calculation. Default is 5.0.
    zw : float, optional
        Z cutoff smoothing parameter. Default is 0.5.
    rc : float, optional
        Cutoff radius for correlation calculation. Default is 6.0.
    width : int, optional
        Width parameter for Raman calculation. Default is 240.
    temperature : float, optional
        Temperature in Kelvin. Default is 330.0.
    M : int, optional
        Number of points for Raman calculation. Default is 20000.
    filter_type : str, optional
        Filter type for Raman calculation. Default is "lorenz".
    save_plots : Optional[List[str]], optional
        List of file paths to save the plots.
        If None, plots are not saved.
    save_data : Optional[str], optional
        Path to save the data.
        If None, data is not saved.

    Returns
    -------
    Tuple of (wavenumber, total_iso, total_aniso, low_range) as np.ndarray.
    """

    weight = cutoff_z(h2o[..., 2], z0, zc, zw)
    weight = np.ones((h2o.shape[0], h2o.shape[1])) - weight

    corr_iso_intra, corr_aniso_intra, corr_iso_inter, corr_aniso_inter = calculate_corr_polar_atomic(
        atomic_polar, weight, h2o, cells, window, rc=rc
    )
    corr_iso_intra = np.sum(corr_iso_intra, axis=1)
    corr_aniso_intra = np.sum(corr_aniso_intra, axis=1)
    corr_iso_inter = np.sum(corr_iso_inter, axis=1)
    corr_aniso_inter = np.sum(corr_aniso_inter, axis=1)

    wavenumber = np.array(
        calculate_raman_atomic(
            corr_iso_intra, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type
        )
    )[0]
    raman_iso_intra = np.array(
        calculate_raman_atomic(
            corr_iso_intra, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type
        )
    )[1]
    raman_aniso_intra = np.array(
        calculate_raman_atomic(
            corr_aniso_intra, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type
        )
    )[1]
    raman_iso_inter = np.array(
        calculate_raman_atomic(
            corr_iso_inter, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type
        )
    )[1]
    raman_aniso_inter = np.array(
        calculate_raman_atomic(
            corr_aniso_inter, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type
        )
    )[1]

    total_iso = raman_iso_intra + raman_iso_inter
    total_aniso = raman_aniso_intra + raman_aniso_inter
    low_range = wavenumber * total_aniso / 1000
    if save_plots is not None:
        # Isotropic Raman
        plt.plot(wavenumber, total_iso, label=r"$H_2O$, iso", scalex=1.5, scaley=2.2)
        plt.xlim((2800, 4000.0))
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("Raman spectra (iso)")
        plt.savefig(save_plots[0], dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

        # Anisotropic Raman
        plt.plot(wavenumber, total_aniso, label=r"$H_2O$, aniso", scalex=1.5, scaley=2.2)
        plt.xlim((2800, 4000.0))
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("Raman spectra (aniso)")
        plt.savefig(save_plots[1], dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

        # Low-frequency Raman
        plt.plot(wavenumber, low_range, label=r"$H_2O$, aniso_low", scalex=1.5, scaley=2.2)
        plt.xlim((0, 2500.0))
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("Low-frequency Raman spectra (aniso)")
        plt.savefig(save_plots[2], dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

    if save_data is not None:
        np.save(save_data, np.array([wavenumber, total_iso]))

    return (
        wavenumber,
        total_iso,
    )


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

    atomic_polar = polar[:, h2o_mask, :, :]

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
    save_data: Optional[str] = None,
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
        plt.plot(ir_range1_xy[:, 0], s_surface_xy, label=r"surface $H_2O$ with s-polarized", scalex=1.5, scaley=2.2)
        plt.plot(ir_range2_xy[:, 0], ir_range2_xy[:, 1], label=r"bulk $H_2O$ with s-polarized", scalex=1.5, scaley=2.2)
        plt.plot(ir_range1_z[:, 0], s_surface_z, label=r"surface $H_2O$ with p-polarized", scalex=1.5, scaley=2.2)
        plt.plot(ir_range2_z[:, 0], ir_range2_z[:, 1], label=r"bulk $H_2O$ with p-polarized", scalex=1.5, scaley=2.2)
        plt.xlim((0, 4000.0))
        plt.xlabel(r"Wavenumber($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"$n(\omega)\alpha(\omega) (10^3 cm^{-1})$", fontdict={"size": 12})
        plt.legend()
        plt.title("IR spectra")
        plt.savefig(save_plot, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

    if save_data is not None:
        # Save columns: wavenumber, surface_xy, bulk_xy, surface_z, bulk_z
        np.savetxt(
            save_data,
            np.column_stack((ir_range1_xy[:, 0], s_surface_xy, ir_range2_xy[:, 1], s_surface_z, ir_range2_z[:, 1])),
        )

    return (ir_range1_xy, ir_range2_xy, ir_range3_xy, ir_range1_z, ir_range2_z, ir_range3_z)


def compute_surface_raman_h2o(
    h2o: np.ndarray,
    oh: np.ndarray,
    cells: np.ndarray,
    atomic_polar_h2o: np.ndarray,
    atomic_polar_oh: np.ndarray,
    dt: float = 0.0005,
    window: int = 2000,
    z0: float = 21.5,
    zc: float = 10.0,
    zw: float = 0.5,
    rc: float = 6.0,
    width: int = 240,
    temperature: float = 330.0,
    M: int = 20000,
    filter_type: str = "lorenz",
    save_plots: Optional[List[str]] = None,
    save_data: Optional[str] = None,
):
    """
    Compute and optionally save the surface Raman spectra (isotropic, anisotropic, and low-frequency)
    from atomic polarizability and geometry data for H2O and OH.

    Parameters
    ----------
    h2o : np.ndarray
        Water molecule coordinates, shape (n_frames, n_h2o, 3).
    oh : np.ndarray
        OH group coordinates, shape (n_frames, n_oh, 3).
    cells : np.ndarray
        Simulation box array, shape (n_frames, 3, 3).
    atomic_polar_h2o : np.ndarray
        Atomic polarizability tensor for H2O, shape (n_frames, n_h2o, 3, 3).
    atomic_polar_oh : np.ndarray
        Atomic polarizability tensor for OH, shape (n_frames, n_oh, 3, 3).
    dt : float, optional
        Time step in picoseconds. Default is 0.0005.
    window : int, optional
        Window size for correlation calculation. Default is 2000.
    z0 : float, optional
        Z cutoff center for weight calculation. Default is 21.5.
    zc : float, optional
        Z cutoff width for weight calculation. Default is 10.0.
    zw : float, optional
        Z cutoff smoothing parameter. Default is 0.5.
    rc : float, optional
        Cutoff radius for correlation calculation. Default is 6.0.
    width : int, optional
        Width parameter for Raman calculation. Default is 240.
    temperature : float, optional
        Temperature in Kelvin. Default is 330.0.
    M : int, optional
        Number of points for Raman calculation. Default is 20000.
    filter_type : str, optional
        Filter type for Raman calculation. Default is "lorenz".
    save_plots : Optional[List[str]], optional
        List of file paths to save the plots: [iso_path, aniso_path, aniso_low_path].
        If None, plots are not saved.
    save_data : Optional[str], optional
        File paths to save the data.
        If None, data is not saved.

    Returns
    -------
    Tuple of (wavenumber, total_iso, total_aniso, low_range) as np.ndarray.
    """

    atomic_polar = np.concatenate([atomic_polar_h2o, atomic_polar_oh], axis=1)
    coords = np.concatenate([h2o, oh], axis=1)

    weight = cutoff_z(coords[..., 2], z0, zc, zw)

    corr_iso_intra, corr_aniso_intra, corr_iso_inter, corr_aniso_inter = calculate_corr_polar_atomic(
        atomic_polar, weight, coords, cells, window, rc=rc
    )
    corr_iso_intra = np.sum(corr_iso_intra[:, : h2o.shape[1]], axis=1)
    corr_aniso_intra = np.sum(corr_aniso_intra[:, : h2o.shape[1]], axis=1)
    corr_iso_inter = np.sum(corr_iso_inter[:, : h2o.shape[1]], axis=1)
    corr_aniso_inter = np.sum(corr_aniso_inter[:, : h2o.shape[1]], axis=1)

    wavenumber = np.array(
        calculate_raman_atomic(
            corr_iso_intra, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type
        )
    )[0]
    raman_iso_intra = np.array(
        calculate_raman_atomic(
            corr_iso_intra, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type
        )
    )[1]
    raman_aniso_intra = np.array(
        calculate_raman_atomic(
            corr_aniso_intra, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type
        )
    )[1]
    raman_iso_inter = np.array(
        calculate_raman_atomic(
            corr_iso_inter, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type
        )
    )[1]
    raman_aniso_inter = np.array(
        calculate_raman_atomic(
            corr_aniso_inter, width=width, dt_ps=dt, temperature=temperature, M=M, filter_type=filter_type
        )
    )[1]

    total_iso = raman_iso_intra + raman_iso_inter
    total_aniso = raman_aniso_intra + raman_aniso_inter
    low_range = wavenumber * total_aniso / 1000

    if save_plots is not None:
        # Isotropic Raman
        plt.plot(wavenumber, total_iso, label=r"$H_2O$, iso", scalex=1.5, scaley=2.2)
        plt.xlim((2800, 4000.0))
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("Raman spectra (iso)")
        plt.savefig(save_plots[0], dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

        # Anisotropic Raman
        plt.plot(wavenumber, total_aniso, label=r"$H_2O$, aniso", scalex=1.5, scaley=2.2)
        plt.xlim((2800, 4000.0))
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("Raman spectra (aniso)")
        plt.savefig(save_plots[1], dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()

        # Low-frequency Raman
        plt.plot(wavenumber, low_range, label=r"$H_2O$, aniso_low", scalex=1.5, scaley=2.2)
        plt.xlim((0, 2500.0))
        plt.xlabel(r"Wavenumber ($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Intensity (a.u.)", fontdict={"size": 12})
        plt.legend()
        plt.title("Low-frequency Raman spectra (aniso)")
        plt.savefig(save_plots[2], dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()
    if save_data is not None:
        np.save(save_data, np.array([wavenumber, total_iso, total_aniso, low_range]))

    return wavenumber, total_iso, total_aniso, low_range


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
    save_data: Optional[str] = None,
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
        weighted_polar[..., 0, 0] + weighted_polar[..., 1, 1], weighted_dipole[..., 2], h2o, window, rc=rc
    )
    sfg = calculate_sfg(corr, width=width, dt_ps=dt, temperature=temperature)

    if save_plot is not None:
        plt.plot(sfg[:, 0], sfg[:, 1], label=r"$H_2O$", scalex=1.5, scaley=2.2)
        plt.xlim((0, 4000.0))
        plt.ylim((-0.12, 0.12))
        plt.xlabel(r"Wavenumber($\rm cm^{-1}$)", fontdict={"size": 12})
        plt.ylabel(r"Im[$\chi^{(2)}$]", fontdict={"size": 12})
        plt.legend()
        plt.title("SFG in xxz and yyz")
        plt.savefig(save_plot, dpi=100, facecolor="white", bbox_inches="tight")
        plt.close()

    if save_data is not None:
        np.savetxt(save_data, sfg)

    return sfg


from ase import Atoms  # noqa: E402
from typing import List, Dict  # noqa: E402


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


def get_lumped_wacent_poses_rel_h2o(stc: Atoms, elem_symbol: str, cutoff: float = 1.0, expected_cn: int = 4):
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
    elem_idx = np.where(stc.symbols == elem_symbol)[0]
    wacent_idx = np.where(stc.symbols == "X")[0]
    elem_poses = stc.positions[elem_idx]
    wacent_poses = stc.positions[wacent_idx]

    cellpar = stc.cell.cellpar()
    assert cellpar is not None
    # dist_matrix
    dist_mat = distance_array(elem_poses, wacent_poses, box=cellpar)

    # each row get distance and select the candidates
    lumped_wacent_poses_rel = []
    for elem_entry, dist_vec in enumerate(dist_mat):
        # print(_elem_idx)
        bool_vec = dist_vec < cutoff
        cn = np.sum(bool_vec)

        # modify neighbor wannier centers coords relative to the center element atom
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
            lumped_wacent_poses_rel = get_lumped_wacent_poses_rel_h2o(
                stc=stc, elem_symbol=elem_symbol, cutoff=1.0, expected_cn=expected_cn
            )
            elem_pos = stc.get_positions()[stc.symbols == elem_symbol]
            X_pos.append(elem_pos + lumped_wacent_poses_rel)

    wfc_pos = np.reshape(X_pos, (len(stc_list), -1))
    return wfc_pos
