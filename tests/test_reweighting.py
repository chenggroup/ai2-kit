from unittest import TestCase, skip
from typing import List
from pathlib import Path

import matplotlib.pyplot as plt
from ase.io import read
from ase import Atom, Atoms

import numpy as np
import dpdata



from ai2_kit.algorithm import reweighting
from ai2_kit.lib import plumed

data_dir = Path(__file__).parent / 'data-sample'


class TestReweighting(TestCase):

    plumed_colvar_file = data_dir / 'plumed_colvar.txt'

    def test_compute_fes(self):
        df = plumed.load_colvar_from_files(self.plumed_colvar_file)
        cvs, bias = plumed.get_cvs_bias_from_df(df, ['d1', 'd2'], 'opes.bias')

        ret = reweighting.compute_fes(cvs, bias, temp=800, grid_size=0.01)

        plt.imshow(np.rot90(ret.fes), cmap=plt.cm.gist_earth, extent=ret.extend)
        plt.colorbar()
        plt.xlabel('d1')
        plt.ylabel('d2')
        plt.title('2D Gaussian KDE')
        plt.show()
