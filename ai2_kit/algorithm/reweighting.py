from typing import Optional, Dict
from collections import namedtuple

from scipy.stats import gaussian_kde
import numpy as np
import json

from ai2_kit.core.util import expand_globs, ensure_dir
from ai2_kit.core.log import get_logger
from ai2_kit.lib import plumed


logger = get_logger(__name__)


default_kB = 0.0083144621  # Boltzmann constant in kJ/mol/K
default_ev_to_kjmol = 96.4853365  # conversion factor from eV to kJ/mol


_FesResult = namedtuple('FesResult', ['fes', 'grid', 'extend'])


def compute_fes(cvs: np.ndarray, bias: np.ndarray, temp: float, grid=None,
                w=1, kB=default_kB, grid_size=100j):
    """
    Compute free energy surface from biased sampling data using Gaussian KDE
    Support 1D and 2D collective variables

    :param cv: collective variable
    :param bias: bias potential
    :param temp: temperature
    :param grid: grid points, if None, will use grid_size to generate automatically
    :param w: weights
    :param kB: Boltzmann constant
    :param grid_size: grid size in np.mgrid style

    :return: free energy
    """
    if cvs.ndim == 2:
        if cvs.shape[0] == 1:
            cvs = cvs[0]  # flatten
        elif cvs.shape[0] != 2:
            raise ValueError(f'Invalid cvs shape {cvs.shape}, only support 1D or 2D cvs')
    elif cvs.ndim != 1:
        raise ValueError(f'Invalid cvs shape {cvs.shape}, only support 1D or 2D cvs')

    kBT = kB * temp
    beta = 1 / kBT
    weights = w * np.exp(beta * bias)
    kde = gaussian_kde(cvs, weights=weights)

    extend = None
    if cvs.ndim == 1:  # handle 1d
        if grid is None:
            x_min, x_max = cvs.min(), cvs.max()
            grid = np.mgrid[x_min:x_max:grid_size]
            extend = [x_min, x_max]
        pdf = kde.evaluate(grid)
    else:  # handle 2d
        if grid is None:
            x_min, x_max = cvs[0].min(), cvs[0].max()
            y_min, y_max = cvs[1].min(), cvs[1].max()
            grid = np.mgrid[x_min:x_max:grid_size, y_min:y_max:grid_size]
            extend = [x_min, x_max, y_min, y_max]
        X, Y = grid
        pos = np.vstack([X.ravel(), Y.ravel()])
        # TODO: generated by copilot, I am not sure if it is correct
        pdf = kde.evaluate(pos).T.reshape(X.shape)
    fes = -kBT * np.log(pdf)
    # return fes with grid and extend for plotting
    return _FesResult(fes=fes, grid=grid, extend=extend)


def compute_kde_weight(baseline_energy: np.ndarray, target_energy: np.ndarray,
                       temp: float, kB=default_kB,
                       ev_to_kjmol=default_ev_to_kjmol,):
    """
    Compute KDE weights for reweighting

    :param baseline_energy: baseline energy in eV, it's the output of DeepMD evaluation
    :param target_energy: target energy in eV, it's the output of DeepMD evaluation
    """
    kBT = kB * temp
    beta = 1 / kBT

    # use relative energy and convert from eV to kJ/mol
    be = (baseline_energy- np.min(baseline_energy)) * ev_to_kjmol
    te = (target_energy- np.min(target_energy)) * ev_to_kjmol
    return np.exp( -beta * (te - be))


def load_dp_energy(*path_or_glob: str):
    """
    load dpdata energy data from npy files

    :param path_or_glob: path or glob pattern to locate data path, for example dpdata/**/energy.npy
    """

    files = expand_globs(*path_or_glob, raise_invalid=True)
    energies = []
    for file in files:
        energy = np.load(file)
        energies.append(energy)

    return np.concatenate(energies)


class ReweightingTool:

    def __init__(self):
        self.data: Dict[str, np.ndarray] = {}
        self.colvar_df = None

    def load_energy(self, *path_or_glob: str, tag: str):
        """
        read baseline data as dpdata.LabeledSystem

        :param path_or_glob: path or glob pattern to locate data path
        :param tag: a string tag to distinguish data, it is suggested to use `baseline` and `target`
        """
        energy = load_dp_energy(path_or_glob)
        if tag in self.data:
            self.data[tag] = np.concatenate([self.data[tag], energy])
        else:
            self.data[tag] = energy
        return self


    def load_colvar(self, *path_or_glob: str):
        """
        load PLUMED COLVAR files and concatenate into a single DataFrame
        you have to ensure the COLVAR data is aligned with energies

        :param path_or_glob: path or glob pattern to locate data path
        """

        paths = expand_globs(path_or_glob)
        self.colvar_df = plumed.load_colvar_from_files(*paths)
        return self

    def reweighting(self, cv: str, bias: str, temp: float,
                    grid_size=0.01, save_fig_to: Optional[str]=None,
                    save_json_to: Optional[str]=None,
                    baseline_tag='baseline', target_tag='target'):
        """
        run reweighting against loaded data

        :param cv: name of collective variable columns, for example d1
        :param bias: name of bias column, for example opes.bias
        :param temp: temperature
        :param save_to: save th
        """
        import matplotlib.pyplot as plt

        cvs_df, bias_df = plumed.get_cvs_bias_from_df(self.colvar_df, [cv], bias)
        logger.info(f'cvs shape: {cvs_df.shape}, bias shape: {bias_df.shape}')

        # compute baseline fes
        baseline_fes = compute_fes(cvs_df, bias_df, temp=temp, grid_size=grid_size)
        grid = baseline_fes.grid

        # compute weight
        baseline_e = self.data[baseline_tag].reshape((-1, 1))
        target_e = self.data[target_tag].reshape((-1, 1))
        logger.info(f'baseline energy shape: {baseline_e.shape}, target energy shape: {target_e.shape}')

        w = compute_kde_weight(baseline_e, target_e, temp=temp)
        w = w.reshape((-1,))

        # compute target fes
        target_fes = compute_fes(cvs_df, bias_df, temp=temp, grid=grid, w=w)

        # use matplot to draw baseline and target 1D FES on the image
        plt.plot(grid, baseline_fes.fes, label='Baseline FES')
        plt.plot(grid, target_fes.fes, label='Target FES')

        plt.xlabel('CV')
        plt.ylabel('Free Energy/$kJ\cdot mol^{-1}$')

        plt.legend()
        fig = plt.gcf()

        if save_json_to is not None:
            data = {
                'baseline_fes': baseline_fes.fes.tolist(),
                'target_fes': target_fes.fes.tolist(),
                'weight': w.tolist(),
                'grid': grid.tolist(),
                'extend': baseline_fes.extend,
            }
            with open(save_json_to, 'w') as f:
                json.dump(data, f)

        if save_fig_to is None:
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            ensure_dir(save_fig_to)
            fig.savefig(save_fig_to, dpi=300, bbox_inches='tight')

