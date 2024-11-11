# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from pathlib import Path
import numpy as np
from scipy import constants

from ai2_kit.domain.dplr import dpdata_read_cp2k_dplr_data, get_sel_ids
from ai2_kit.domain.dpff import dpdata_read_cp2k_dpff_data


class CP2kTestData:
    def __init__(self):
        # both dplr and dpff
        self.cp2k_dir = str(
            Path(__file__).parent / "data-sample/cp2k_nacl_finite_field"
        )
        self.cp2k_output = "output"
        self.wannier_file = "wannier.xyz"
        self.type_map = ["O", "H", "Na", "Cl"]
        self.sys_charge_map = [6.0, 1.0, 9.0, 7.0]
        self.model_charge_map = [-8.0, -8.0, -8.0]
        self.sel_type = [0, 2, 3]

        # dpff only
        self.ewald_h = 0.5
        self.ewald_beta = 0.4
        # E-field: a.u. to V/angstrom
        coeff = (
            constants.physical_constants["atomic unit of electric field"][0]
            * constants.angstrom
        )
        self.ext_efield = [0.0, 0.0, 0.0002 * coeff]


class TestDPLRDPData(unittest.TestCase, CP2kTestData):
    def setUp(self):
        CP2kTestData.__init__(self)
        backend = "tf"
        self.tf_data = dpdata_read_cp2k_dplr_data(
            self.cp2k_dir,
            self.cp2k_output,
            self.wannier_file,
            self.type_map,
            self.sys_charge_map,
            self.model_charge_map,
            self.sel_type,
            backend=backend,
        )

        backend = "pt"
        self.pt_data = dpdata_read_cp2k_dplr_data(
            self.cp2k_dir,
            self.cp2k_output,
            self.wannier_file,
            self.type_map,
            self.sys_charge_map,
            self.model_charge_map,
            self.sel_type,
            backend=backend,
        )
        self.sel_ids = get_sel_ids(self.pt_data, self.type_map, self.sel_type)

    def test_shape(self):
        natoms = self.pt_data.get_natoms()

        assert self.pt_data.data["atomic_dipole"].shape[1] == natoms * 3
        assert self.tf_data.data["atomic_dipole"].shape[1] == len(self.sel_ids) * 3

    def test_consistent(self):
        atomic_dipole_tf_reshape = self.tf_data.data["atomic_dipole"].reshape(-1, 3)
        atomic_dipole_pt_reshape = self.pt_data.data["atomic_dipole"].reshape(-1, 3)
        diff = atomic_dipole_pt_reshape[self.sel_ids] - atomic_dipole_tf_reshape
        self.assertTrue(np.max(np.abs(diff)) < 1e-6)


class TestDPFFDPData(TestDPLRDPData):
    def setUp(self):
        CP2kTestData.__init__(self)
        backend = "tf"
        self.tf_data = dpdata_read_cp2k_dpff_data(
            self.cp2k_dir,
            self.cp2k_output,
            self.wannier_file,
            self.type_map,
            self.sys_charge_map,
            self.model_charge_map,
            self.ewald_h,
            self.ewald_beta,
            self.ext_efield,
            self.sel_type,
            backend=backend,
        )

        backend = "pt"
        self.pt_data = dpdata_read_cp2k_dpff_data(
            self.cp2k_dir,
            self.cp2k_output,
            self.wannier_file,
            self.type_map,
            self.sys_charge_map,
            self.model_charge_map,
            self.ewald_h,
            self.ewald_beta,
            self.ext_efield,
            self.sel_type,
            backend=backend,
        )
        self.sel_ids = get_sel_ids(self.pt_data, self.type_map, self.sel_type)

    def test_numerical(self):
        pass


if __name__ == "__main__":
    unittest.main()
