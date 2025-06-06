import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error
from scipy.integrate import trapz
from typing import Sequence, Optional



def plot_energy_force_error(
        ener_neu: np.ndarray,
        ener_red: np.ndarray,
        frcs_neu: np.ndarray,
        frcs_red: np.ndarray,
        n_atoms: int = 191,
        save_plot: Optional[str] = None
):
    """
    Plot and analyze the error between DFT and MLP for energy and force, for both neutral and reduced systems.

    Parameters
    ----------
    ener_neu : np.ndarray
        Array of shape (N, 2), neutral system energies. Column 0: DFT, Column 1: MLP.
    ener_red : np.ndarray
        Array of shape (N, 2), reduced system energies. Column 0: DFT, Column 1: MLP.
    frcs_neu : np.ndarray
        Array of shape (N, 6), neutral system forces. Columns 0-2: DFT, Columns 3-5: MLP.
    frcs_red : np.ndarray
        Array of shape (N, 6), reduced system forces. Columns 0-2: DFT, Columns 3-5: MLP.
    n_atoms : int, default=191
        Number of atoms for normalization.
    save_plot : Optional[str], default=None
        If not None, save the figure to this path.

    Returns
    -------
    None
        The function creates and optionally saves or shows a figure with two subplots:
        (1) DFT vs. MLP energy scatter and RMSE/STD annotations.
        (2) DFT vs. MLP force scatter and RMSE/STD annotations.
    """
    lw2 = 2
    fs =18
    energy_scale = 0.04
    force_scale = 10
    alpha_red_energy = 0.1
    alpha_red_force = 0.2

    figure_1 = plt.figure(figsize=(10, 3), dpi=200)

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.3)

    # Energy subplot
    sp_1 = plt.subplot(gs1[0, 0])
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(lw2)

    mean_x = np.mean(ener_neu[:, 0])
    ener_fp_neu = (ener_neu[:, 0]-mean_x)/n_atoms
    ener_ml_neu = (ener_neu[:, 1]-mean_x)/n_atoms
    mean_x = np.mean(ener_red[:, 0])
    ener_fp_red = (ener_red[:, 0]-mean_x)/n_atoms
    ener_ml_red = (ener_red[:, 1]-mean_x)/n_atoms

    plt.plot(ener_fp_neu, ener_ml_neu, "b.")
    plt.plot(ener_fp_red, ener_ml_red, "r.", alpha=alpha_red_energy)

    plt.xlim(-energy_scale, energy_scale)
    plt.ylim(-energy_scale, energy_scale)
    plt.xticks(np.arange(-energy_scale, energy_scale + 0.001, energy_scale / 2), fontsize=fs - 6)
    plt.yticks(np.arange(-energy_scale, energy_scale + 0.001, energy_scale / 2), fontsize=fs - 6)
    plt.plot([-0.1, 0.1], [-0.1, 0.1], "k--")

    # RMSE and STD for neutral
    x = ener_fp_neu
    y = ener_ml_neu
    rmse = np.sqrt(mean_squared_error(x, y))
    std = np.sqrt(np.var((x - y) ** 2))
    plt.text(-0.020 / 0.045 * energy_scale, -0.040 / 0.045 * energy_scale,
             "RMSE:\n" + r" %4.2e $\pm$ %4.2e" % (rmse, std),
             fontsize=fs - 4, color="navy")

    # RMSE and STD for reduced
    x = ener_fp_red
    y = ener_ml_red
    rmse = np.sqrt(mean_squared_error(x, y))
    std = np.sqrt(np.var((x - y) ** 2))
    plt.text(-0.040 / 0.045 * energy_scale, 0.025 / 0.045 * energy_scale,
             "RMSE:\n" + r" %4.2e $\pm$ %4.2e" % (rmse, std),
             fontsize=fs - 4, color="red")

    plt.ylabel(r"rel. $\rm E_{MLP}$ (eV/atom)", fontsize=fs)
    plt.xlabel(r"rel. $\rm E_{DFT}$ (eV/atom)", fontsize=fs)

    # Force subplot
    sp_1 = plt.subplot(gs1[0, 1])
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(lw2)

    frcs_fp_neu = np.ravel(frcs_neu[:, 0:3])
    frcs_ml_neu = np.ravel(frcs_neu[:, 3:6])
    frcs_fp_red = np.ravel(frcs_red[:, 0:3])
    frcs_ml_red = np.ravel(frcs_red[:, 3:6])

    plt.plot(frcs_fp_neu, frcs_ml_neu, ".b")
    plt.plot(frcs_fp_red, frcs_ml_red, ".r", alpha=alpha_red_force)

    plt.xlim(-force_scale, force_scale)
    plt.ylim(-force_scale, force_scale)
    plt.xticks(np.arange(-force_scale, force_scale + 0.001, force_scale / 2), fontsize=fs - 6)
    plt.yticks(np.arange(-force_scale, force_scale + 0.001, force_scale / 2), fontsize=fs - 6)
    plt.plot([-force_scale, force_scale], [-force_scale, force_scale], "k--")

    plt.ylabel(r"$\rm F_{MLP}$ (eV/$\rm \AA$)", fontsize=fs)
    plt.xlabel(r"$\rm F^i_{DFT}$ (eV/$\rm \AA$)", fontsize=fs)

    # RMSE and STD for neutral
    x = frcs_fp_neu
    y = frcs_ml_neu
    rmse = np.sqrt(mean_squared_error(x, y))
    std = np.sqrt(np.var((x - y) ** 2))
    plt.text(-0.020 / 0.045 * force_scale, -0.040 / 0.045 * force_scale,
             "RMSE:\n" + r" %4.2e $\pm$ %4.2e" % (rmse, std),
             fontsize=fs - 4, color="navy")

    # RMSE and STD for reduced
    x = frcs_fp_red
    y = frcs_ml_red
    rmse = np.sqrt(mean_squared_error(x, y))
    std = np.sqrt(np.var((x - y) ** 2))
    plt.text(-0.040 / 0.045 * force_scale, 0.025 / 0.045 * force_scale,
             "RMSE:\n" + r" %4.2e $\pm$ %4.2e" % (rmse, std),
             fontsize=fs - 4, color="red")

    if save_plot is not None:
        plt.savefig(save_plot, bbox_inches='tight')
    plt.close()



def plot_ti_results(
    etas: Sequence[float],
    vegs: Sequence[float],
    etas_fp: Sequence[float],
    vegs_fp: Sequence[float],
    save_plot: Optional[str] = None
):
    """
    Plot thermodynamic integration (TI) results for MLMD and FP data, and compute integrals.

    Parameters
    ----------
    etas : Sequence[float]
        List of lambda/eta values for MLMD.
    vegs : Sequence[float]
        List of <Delta E> values for MLMD, same length as etas.
    etas_fp : Sequence[float]
        List of lambda/eta values for FP.
    vegs_fp : Sequence[float]
        List of <Delta E> values for FP, same length as etas_fp.
    save_plot : Optional[str], default=None
        If not None, save the figure to this path.

    Returns
    -------
    (float, float)
        Tuple of (MLMD integral, FP integral).
    """
    lw2 = 2
    fs = 18
    y_ticks = (0, 1, 2, 3)
    ylim = (-0.05, 3.0)
    mlmd_color = "navy"
    fp_color = "r"
    mlmd_line = "-o"
    fp_line = "r--x"
    mlmd_markeredgecolor = "k"

    figure_1 = plt.figure(figsize=(10, 3), dpi=200)
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.3)

    # Subplot 1: time evolution (not used in this function, but kept for compatibility)
    sp_1 = plt.subplot(gs1[0, 0])
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(lw2)

    plt.ylabel(r'$\langle\Delta E\rangle_{\eta}$ / $\rm eV$', fontsize=fs)
    plt.xlabel(r't / ns', fontsize=fs)
    plt.xticks(fontsize=fs - 4)
    plt.yticks(y_ticks, fontsize=fs - 4)
    plt.ylim(*ylim)

    # Subplot 2: TI curve
    sp_1 = plt.subplot(gs1[0, 1])
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(lw2)

    plt.plot(etas, vegs, mlmd_line, color=mlmd_color, markeredgecolor=mlmd_markeredgecolor)
    plt.plot(etas_fp, vegs_fp, fp_line)
    plt.xlabel(r'$\eta$', fontsize=fs)
    plt.ylabel(r'$\langle\Delta E\rangle_{\eta}$ / $\rm eV$', fontsize=fs)
    plt.ylim(*ylim)
    plt.xticks(fontsize=fs - 4)
    plt.yticks(y_ticks, fontsize=fs - 4)

    integral = trapz(vegs, etas)
    integral_fp = trapz(vegs_fp, etas_fp)

    plt.text(1.00, 0.8, " MLMD: %5.3f eV" % (integral), color=mlmd_color, fontsize=fs - 4, horizontalalignment="right")
    plt.text(1.00, 0.4, " BLYP-D3: %5.3f eV" % (integral_fp), color=fp_color, fontsize=fs - 4, horizontalalignment="right")

    if save_plot is not None:
        plt.savefig(save_plot, bbox_inches='tight')
    plt.close()
    return integral, integral_fp


def plot_vertical_energy_gaps(
    gap_data: pd.DataFrame,
    save_plot: Optional[str] = None
):
    """
    Plot vertical energy gaps for multiple columns in a DataFrame.

    Parameters
    ----------
    gap_data : pd.DataFrame
        DataFrame with a 'time' column and one or more energy gap columns.
    save_plot : Optional[str], default=None
        If not None, save the figure to this path.

    Returns
    -------
    None
    """
    # plt magic number
    figsize = (10, 6)
    dpi = 500
    legend_loc = 'upper right'
    legend_bbox = (1.25, 1)
    xlabel = 't/ps'
    ylabel = r'$\langle \Delta E\rangle_\eta$ / eV'
    title = 'vertical energy gaps'
    ylim = (0, 3)
    xlim = (0, 1050)
    yticks = np.linspace(0, 3, 4)

    plt.figure(figsize=figsize, dpi=dpi)
    for column in gap_data.columns[1:]:
        plt.plot(gap_data['time'], gap_data[column], label=column, antialiased=True)
    plt.legend(loc=legend_loc, bbox_to_anchor=legend_bbox)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    plt.yticks(yticks)
    if save_plot is not None:
        plt.savefig(save_plot, bbox_inches='tight')
    plt.close()



def plot_smoothed_vertical_energy_gaps(
    gap_data: pd.DataFrame,
    save_plot: Optional[str] = None
):
    """
    Plot smoothed vertical energy gaps for multiple columns in a DataFrame using moving average.

    Parameters
    ----------
    gap_data : pd.DataFrame
        DataFrame with a 'time' column and one or more energy gap columns.
    save_plot : Optional[str], default=None
        If not None, save the figure to this path.

    Returns
    -------
    None
    """
    # Plotting magic numbers
    figsize = (10, 6)
    dpi = 500
    legend_loc = 'upper right'
    legend_bbox = (1.25, 1)
    xlabel = 't/ps'
    ylabel = r'$\langle \Delta E\rangle_\eta$ / eV'
    title = 'mean vertical energy gaps'
    ylim = (0, 3)
    xlim = (0, 1050)
    yticks = np.linspace(0, 3, 4)
    window = 20000
    min_periods = 18

    plt.figure(figsize=figsize, dpi=dpi)
    for column in gap_data.columns[1:]:
        smoothed_data = gap_data[column].rolling(window=window, center=False, min_periods=min_periods).mean()
        plt.plot(gap_data['time'], smoothed_data, label=column + ' (mean)', antialiased=False)
    plt.legend(loc=legend_loc, bbox_to_anchor=legend_bbox)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    plt.yticks(yticks)
    if save_plot is not None:
        plt.savefig(save_plot, bbox_inches='tight')
    plt.close()



def plot_thermodynamic_integration(
    mean_data: pd.DataFrame,
    save_plot: Optional[str] = None
):
    """
    Plot thermodynamic integration curve from mean energy gap data.

    Parameters
    ----------
    mean_data : pd.DataFrame
        DataFrame with columns 'eta' and 'Mean'.
    save_plot : Optional[str], default=None
        If not None, save the figure to this path.

    Returns
    -------
    None
    """
    # Plotting magic numbers
    figsize = (10, 6)
    dpi = 500
    marker = 'o'
    linestyle = '-'
    color = 'blue'
    xlabel = r'$\eta$'
    ylabel = r'$\langle \Delta E\rangle_\eta$ / eV'
    title = 'thermodynamic integration'
    ylim = (0, 3)
    xlim = (0, 1)
    xticks = np.linspace(0, 1, 5)
    yticks = np.linspace(0, 3, 4)

    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(mean_data['eta'], mean_data['Mean'], marker=marker, linestyle=linestyle, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    plt.xticks(xticks)
    plt.yticks(yticks)
    if save_plot is not None:
        plt.savefig(save_plot, bbox_inches='tight')
    plt.close()
