import sys
# for path in sys.path:
#     print(path)

from pathlib import Path

current_dir = str(Path.cwd())
sys.path.append(current_dir)

import unittest  # noqa: E402

import numpy as np  # noqa: E402
import dpdata  # noqa: E402
from ase.io import read  # noqa: E402

from ai2_kit.feat.spectrum import md_spectra  # noqa: E402

sample_dir = Path(__file__).parent / "data-sample" / "md_spectra_sample"
output_dir = Path(__file__).parent / "data-sample" / "md_spectra_output"
FAST_FRAMES = 1024
FAST_WINDOW = 256
FAST_M = 2048


def _load_fast_array(path: Path, nframes: int = FAST_FRAMES) -> np.ndarray:
    return np.load(path)[:nframes]


def _load_fast_traj(nframes: int = FAST_FRAMES):
    traj = dpdata.System(sample_dir / "traj", fmt="deepmd/npy")
    return traj.sub_system(list(range(nframes)))


class TestMdSpectra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        output_dir.mkdir(parents=True, exist_ok=True)

    def _assert_valid_curve(self, x: np.ndarray, y: np.ndarray):
        self.assertEqual(x.ndim, 1)
        self.assertEqual(y.ndim, 1)
        self.assertEqual(x.shape, y.shape)
        self.assertGreater(x.size, 10)
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertTrue(np.all(np.diff(x) >= 0))

    def test_extract_atomic_polar_from_traj_h2o(self):
        # corresponds to file cal_polar_wan.py
        traj = _load_fast_traj()
        polar: np.ndarray = _load_fast_array(sample_dir / "wannier_polar.npy")
        name = "atomic_polar_wan.npy"

        md_spectra.extract_atomic_polar_from_traj_h2o(
            traj=traj,
            polar=polar,
            type_O=1,
            type_H=2,
            r_bond=1.3,
            save_data=output_dir / name,
        )

        np.testing.assert_array_equal(_load_fast_array(sample_dir / name), np.load(output_dir / name))

    def test_compute_atomic_dipole_h2o(self):
        # corresponds to file cal_dipole_wan.py
        traj = _load_fast_traj()
        wannier: np.ndarray = _load_fast_array(sample_dir / "wannier_dipole.npy")
        wannier = wannier.reshape(traj.get_nframes(), -1, 3)
        name_h2o = "h2o.npy"
        name_adw = "atomic_dipole_wan.npy"

        md_spectra.compute_atomic_dipole_h2o(
            traj=traj,
            wannier=wannier,
            type_O=1,
            type_H=2,
            r_bond=1.3,
            save_datas=[output_dir / name_h2o, output_dir / name_adw],
        )

        h2o_out = np.load(output_dir / name_h2o)
        atomic_dipole_out = np.load(output_dir / name_adw)
        self.assertEqual(h2o_out.shape[0], FAST_FRAMES)
        self.assertEqual(h2o_out.shape[2], 3)
        self.assertGreater(h2o_out.shape[1], 0)
        self.assertEqual(atomic_dipole_out.shape, h2o_out.shape)
        self.assertTrue(np.all(np.isfinite(h2o_out)))
        self.assertTrue(np.all(np.isfinite(atomic_dipole_out)))

    def test_compute_surface_ir_spectra_h2o(self):
        dt = 0.0005
        window = FAST_WINDOW

        h2o = _load_fast_array(sample_dir / "h2o.npy")
        cells = _load_fast_array(sample_dir / "traj/set.000/box.npy").reshape(h2o.shape[0], 3, 3)
        atomic_dipole = _load_fast_array(sample_dir / "atomic_dipole_wan.npy")
        atomic_dipole = atomic_dipole.reshape(atomic_dipole.shape[0], -1, 3)
        name = "ir_sp.dat"

        wavenumber, ir_h2o = md_spectra.compute_surface_ir_spectra_h2o(
            h2o=h2o,
            cells=cells,
            atomic_dipole=atomic_dipole,
            dt=dt,
            window=window,
            width=25,
            temperature=330.0,
            M=FAST_M,
            save_plot=output_dir / "ir_sp.png",
            save_data=output_dir / name,
        )

        self._assert_valid_curve(wavenumber, ir_h2o)
        self.assertTrue((output_dir / name).exists())
        self.assertTrue((output_dir / "ir_sp.png").exists())
        self.assertEqual(np.loadtxt(output_dir / name).shape[1], 2)

    def test_compute_surface_sfg_h2o(self):
        dt = 0.0005
        window = FAST_WINDOW

        h2o = _load_fast_array(sample_dir / "h2o.npy")
        cells = _load_fast_array(sample_dir / "traj/set.000/box.npy").reshape(h2o.shape[0], 3, 3)
        atomic_dipole = _load_fast_array(sample_dir / "atomic_dipole_wan.npy").reshape(h2o.shape[0], -1, 3)
        atomic_polar = _load_fast_array(sample_dir / "atomic_polar_wan.npy").reshape(h2o.shape[0], -1, 3, 3)
        name = "SFG.dat"

        wavenumber, sfg_imag = md_spectra.compute_surface_sfg_h2o(
            h2o=h2o,
            cells=cells,
            atomic_dipole=atomic_dipole,
            atomic_polar=atomic_polar,
            dt=dt,
            window=window,
            z0=22.5,
            zc=2.5,
            zw=2.6,
            rc=6.75,
            width=50,
            temperature=330.0,
            M=FAST_M,
            save_plot=output_dir / "sfg.png",
            save_data=output_dir / name,
        )

        self._assert_valid_curve(wavenumber, sfg_imag)
        self.assertTrue((output_dir / name).exists())
        self.assertTrue((output_dir / "sfg.png").exists())
        self.assertEqual(np.loadtxt(output_dir / name).shape[1], 2)

    def test_compute_bulk_ir_h2o(self):
        dt = 0.0005
        window = FAST_WINDOW

        h2o = _load_fast_array(sample_dir / "traj/set.000/h2o.npy")
        cells = _load_fast_array(sample_dir / "traj/set.000/box.npy").reshape(h2o.shape[0], 3, 3)
        atomic_dipole = _load_fast_array(sample_dir / "traj/set.000/atomic_dipole_wan_h2o.npy").reshape(h2o.shape[0], -1, 3)
        name = "ir_spectrum.npy"

        wavenumber, ir = md_spectra.compute_bulk_ir_h2o(
            h2o=h2o,
            cells=cells,
            atomic_dipole=atomic_dipole,
            dt=dt,
            window=window,
            z0=21.5,
            zc=5.0,
            zw=0.5,
            rc=6.0,
            width=240,
            temperature=330.0,
            M=FAST_M,
            filter_type="lorenz",
            save_plot=output_dir / "bulk_ir.png",
            save_data=output_dir / name,
        )

        self._assert_valid_curve(wavenumber, ir)
        saved = np.loadtxt(output_dir / name)
        self.assertEqual(saved.shape[1], 2)
        self.assertEqual(saved.shape[0], wavenumber.shape[0])

    def test_compute_bulk_raman_h2o(self):
        dt = 0.0005
        window = FAST_WINDOW

        h2o = _load_fast_array(sample_dir / "traj/set.000/h2o.npy")
        cells = _load_fast_array(sample_dir / "traj/set.000/box.npy").reshape(h2o.shape[0], 3, 3)
        atomic_polar = _load_fast_array(sample_dir / "traj/set.000/atomic_polar_wan_h2o.npy").reshape(h2o.shape[0], -1, 3, 3)
        name = "br.npy"

        wavenumber, total_iso, total_aniso, low_range = md_spectra.compute_bulk_raman_h2o(
            h2o=h2o,
            cells=cells,
            atomic_polar=atomic_polar,
            dt=dt,
            window=window,
            z0=21.5,
            zc=5.0,
            zw=0.5,
            rc=6.0,
            width=240,
            temperature=330.0,
            M=FAST_M,
            filter_type="lorenz",
            save_plots=[
                output_dir / "bulk_raman_iso.png",
                output_dir / "bulk_raman_aniso.png",
                output_dir / "bulk_raman_aniso_low.png",
            ],
            save_data=output_dir / name,
        )

        self._assert_valid_curve(wavenumber, total_iso)
        self._assert_valid_curve(wavenumber, total_aniso)
        self._assert_valid_curve(wavenumber, low_range)
        saved = np.loadtxt(output_dir / name)
        self.assertEqual(saved.shape[1], 4)
        self.assertEqual(saved.shape[0], wavenumber.shape[0])

    def test_compute_surface_raman_h2o(self):
        dt = 0.0005
        window = FAST_WINDOW

        h2o = _load_fast_array(sample_dir / "traj/set.000/h2o.npy")
        cells = _load_fast_array(sample_dir / "traj/set.000/box.npy").reshape(h2o.shape[0], 3, 3)
        atomic_polar = _load_fast_array(sample_dir / "traj/set.000/atomic_polar_wan_h2o.npy").reshape(h2o.shape[0], -1, 3, 3)
        name = "sr.npy"

        wavenumber, total_iso, total_aniso, low_range = md_spectra.compute_surface_raman_h2o(
            h2o=h2o,
            cells=cells,
            atomic_polar=atomic_polar,
            dt=dt,
            window=window,
            z0=25.0,
            zc=7.5,
            zw=0.5,
            rc=6.0,
            width=240,
            temperature=330.0,
            M=FAST_M,
            filter_type="lorenz",
            save_plots=[
                output_dir / "sur_raman_iso.png",
                output_dir / "sur_raman_aniso.png",
                output_dir / "sur_raman_aniso_low.png",
            ],
            save_data=output_dir / name,
        )

        self._assert_valid_curve(wavenumber, total_iso)
        self._assert_valid_curve(wavenumber, total_aniso)
        self._assert_valid_curve(wavenumber, low_range)
        saved = np.loadtxt(output_dir / name)
        self.assertEqual(saved.shape[1], 4)
        self.assertEqual(saved.shape[0], wavenumber.shape[0])

    def test_set_lumped_wfc_h2o(self):
        names = ["coord.npy", "box.npy", "atomic_dipole.npy"]
        stc_list_file = sample_dir / "wannier.xyz"
        a = b = c = 12.42
        stc_list = read(stc_list_file, index=":")
        atom_pos = []
        box = np.zeros((len(stc_list), 9))
        box[:, 0] = a
        box[:, 4] = b
        box[:, 8] = c

        for stc in stc_list:
            for atom in stc:
                if atom.symbol == "O" or atom.symbol == "H":
                    atom_pos.append(atom.position)
        convert_coord = np.reshape(atom_pos, (len(stc_list), -1))
        np.save(output_dir / names[0], convert_coord)
        np.save(output_dir / names[1], box)

        cell = [a, b, c]
        stc_list = md_spectra.set_cells_h2o(stc_list, cell)

        lumped_dict = {"O": 4}
        wfc_pos = md_spectra.set_lumped_wfc_h2o(stc_list, lumped_dict)
        np.save(output_dir / names[2], wfc_pos)
        for name in names:
            np.testing.assert_array_equal(np.load(sample_dir / name), np.load(output_dir / name))


if __name__ == "__main__":
    unittest.main()

    # def test_specific(test_name: str):
    #     suite = unittest.TestSuite()
    #     suite.addTest(TestMdSpectra(test_name))
    #     runner = unittest.TextTestRunner()
    #     runner.run(suite)

    # test_specific(
    #     "test_set_lumped_wfc",
    # )
