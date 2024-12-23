# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import glob
import shutil

import unittest
from pathlib import Path
import numpy as np
from scipy import constants

from ai2_kit.domain.dplr import dpdata_read_cp2k_dplr_data, get_sel_ids
from ai2_kit.domain.dpff import dpdata_read_cp2k_dpff_data
from ai2_kit.tool.dpdata import DpdataTool


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
        self.wannier_file_with_ion = "wannier_with_ion.xyz"

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

        self.data = dpdata_read_cp2k_dplr_data(
            self.cp2k_dir,
            self.cp2k_output,
            self.wannier_file,
            self.type_map,
            self.sel_type,
        )
        self.data_with_ion = dpdata_read_cp2k_dplr_data(
            self.cp2k_dir,
            self.cp2k_output,
            self.wannier_file_with_ion,
            self.type_map,
            self.sel_type,
        )
        self.sel_ids = get_sel_ids(self.data, self.type_map, self.sel_type)

    def test_shape(self):
        natoms = self.data.get_natoms()

        assert self.data.data["atomic_dipole"].shape[1] == natoms
        assert self.data.data["atomic_dipole"].shape[2] == 3

    def test_consistent(self):
        np.testing.assert_allclose(
            self.data.data["atomic_dipole"][0],
            self.data_with_ion.data["atomic_dipole"][0],
        )


class TestDPFFDPData(TestDPLRDPData):
    def setUp(self):
        CP2kTestData.__init__(self)

        self.data = dpdata_read_cp2k_dpff_data(
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
        )
        self.sel_ids = get_sel_ids(self.data, self.type_map, self.sel_type)


class TestWriteDPData(unittest.TestCase):
    def setUp(self) -> None:
        self.work_dir = str(
            Path(__file__).parent / "data-sample/cp2k_wannier_localization"
        )
        self.type_map = ["O", "H", "K", "F", "Pt"]
        self.sel_type = [0, 2, 3]

        self.data_1 = dpdata_read_cp2k_dplr_data(
            cp2k_dir="%s/task.000" % self.work_dir,
            cp2k_output="output",
            wannier_file="wannier.xyz",
            type_map=self.type_map,
            sel_type=self.sel_type,
            wannier_spread_file="wannier_spread.out",
        )
        self.data_2 = dpdata_read_cp2k_dplr_data(
            cp2k_dir="%s/task.001" % self.work_dir,
            cp2k_output="output",
            wannier_file="wannier.xyz",
            type_map=self.type_map,
            sel_type=self.sel_type,
            wannier_spread_file="wannier_spread.out",
        )
        self.sel_ids = get_sel_ids(self.data_1, self.type_map, self.sel_type)

    def test_single_consistent(self):
        try:
            shutil.rmtree(".tmp_data-v2")
            shutil.rmtree(".tmp_data-v3")
        except FileNotFoundError:
            pass

        obj = DpdataTool(systems=[self.data_1])
        obj.write(".tmp_data-v2", v2=True, sel_symbol=["O", "K", "F"])
        obj.write(".tmp_data-v3", v2=False)

        atomic_dipole_v2 = np.load(
            glob.glob(".tmp_data-v2/**/atomic_dipole.npy", recursive=True)[0]
        ).reshape([-1, 3])
        atomic_dipole_v3 = np.load(
            glob.glob(".tmp_data-v3/**/atomic_dipole.npy", recursive=True)[0]
        ).reshape([-1, 3])

        diff = atomic_dipole_v3[self.sel_ids] - atomic_dipole_v2
        self.assertTrue(np.max(np.abs(diff)) < 1e-6)

        shutil.rmtree(".tmp_data-v2")
        shutil.rmtree(".tmp_data-v3")

    def test_merged_consistent(self):
        try:
            shutil.rmtree(".tmp_data-single")
            shutil.rmtree(".tmp_data-merged")
        except FileNotFoundError:
            pass

        # data_1 and data_2 have the same composition but different order
        obj_single = DpdataTool(systems=[self.data_2])
        obj_merged = DpdataTool(systems=[self.data_1, self.data_2])
        obj_single.write(".tmp_data-single")
        obj_merged.write(".tmp_data-merged")

        dname_single = glob.glob(".tmp_data-single/*/")[0]
        dname_merged = glob.glob(".tmp_data-merged/*/")[0]

        coord_merged = np.load(os.path.join(dname_merged, "set.000/coord.npy"))
        type_map_merged = np.loadtxt(
            os.path.join(dname_merged, "type_map.raw"), dtype=str
        )

        # data with change in order
        coord_test = np.load(os.path.join(dname_single, "set.000/coord.npy"))
        atype = np.loadtxt(os.path.join(dname_single, "type.raw"), dtype=int)
        type_map = np.loadtxt(
            os.path.join(dname_single, "type_map.raw"),
            dtype=str,
        )
        symbols = type_map[atype]

        # sort id in type_map_merged
        sorted_id = [np.where(symbols == s)[0] for s in type_map_merged]
        sorted_id = np.concatenate(sorted_id)
        # print(symbols[sorted_id])
        # check if the data is the same
        np.testing.assert_allclose(
            coord_merged[1].reshape([-1, 3]), coord_test.reshape([-1, 3])[sorted_id]
        )

        wc_merged = np.load(os.path.join(dname_merged, "set.000/atomic_dipole.npy"))
        wc_test = np.load(os.path.join(dname_single, "set.000/atomic_dipole.npy"))
        np.testing.assert_allclose(
            wc_merged[1].reshape([-1, 3]), wc_test.reshape([-1, 3])[sorted_id]
        )

        wc_spread_merged = np.load(
            os.path.join(dname_merged, "set.000/wannier_spread.npy")
        )
        wc_spread_test = np.load(
            os.path.join(dname_single, "set.000/wannier_spread.npy")
        )
        np.testing.assert_allclose(
            wc_spread_merged[1].reshape([-1, 4]),
            wc_spread_test.reshape([-1, 4])[sorted_id],
        )

        shutil.rmtree(".tmp_data-single")
        shutil.rmtree(".tmp_data-merged")


class TestDPLRSorted(unittest.TestCase):
    def setUp(self) -> None:
        self.cp2k_output = "output"
        self.wannier_file = "wannier.xyz"
        self.type_map = [
            "Na",
            "S",
            "O",
            "N",
            "Cl",
            "H",
        ]
        self.sel_type = [0, 2]

        self.data_sorted = dpdata_read_cp2k_dplr_data(
            str(Path(__file__).parent / "data-sample/cp2k_wannier_sort_test/sorted"),
            self.cp2k_output,
            self.wannier_file,
            self.type_map,
            self.sel_type,
        )
        self.data_random = dpdata_read_cp2k_dplr_data(
            str(Path(__file__).parent / "data-sample/cp2k_wannier_sort_test/random"),
            self.cp2k_output,
            self.wannier_file,
            self.type_map,
            self.sel_type,
        )

        self.random_ids = np.loadtxt(
            str(
                Path(__file__).parent
                / "data-sample/cp2k_wannier_sort_test/random_ids.txt"
            ),
            dtype=int,
        )

    def test_coord(self):
        np.testing.assert_allclose(
            self.data_sorted.data["coords"][0][self.random_ids],
            self.data_random.data["coords"][0],
        )

    def test_consistent(self):
        np.testing.assert_allclose(
            self.data_sorted.data["atomic_dipole"][0][self.random_ids],
            self.data_random.data["atomic_dipole"][0],
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
