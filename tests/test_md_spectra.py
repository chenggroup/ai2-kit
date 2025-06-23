import sys
# for path in sys.path:
#     print(path)

from pathlib import Path

current_dir = str(Path.cwd())
sys.path.append(current_dir)

import unittest  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import dpdata  # noqa: E402

from ai2_kit.feat.spectrum import md_spectra  # noqa: E402

sample_dir = Path(__file__).parent / "data-sample" / "md_spectra_sample"
output_dir = Path(__file__).parent / "data-sample" / "md_spectra_output"


class TestMdSpectra(unittest.TestCase):
    def test_extract_atomic_polar_from_traj_h2o(self):
        # corresponds to file cal_polar_wan.py
        traj = dpdata.System(sample_dir / "traj", fmt="deepmd/npy")
        polar: np.ndarray = np.load(sample_dir / "wannier_polar.npy")
        polar = -polar.reshape(polar.shape[0], -1, 3, 3)
        name = "atomic_polar_wan.npy"

        md_spectra.extract_atomic_polar_from_traj_h2o(
            traj=traj,
            polar=polar,
            type_O=1,
            type_H=2,
            r_bond=1.3,
            save_data=output_dir / name,
        )

        np.testing.assert_array_equal(np.load(sample_dir / name), np.load(output_dir / name))

    def test_compute_atomic_dipole_h2o(self):
        # corresponds to file cal_dipole_wan.py
        a0 = 0.52917721067
        traj = dpdata.System(sample_dir / "traj", fmt="deepmd/npy")
        wannier: np.ndarray = np.load(sample_dir / "wannier_dipole.npy")
        wannier = wannier.reshape(traj.get_nframes(), -1, 3)
        name_h2o = "h2o.npy"
        name_adw = "atomic_dipole_wan.npy"

        md_spectra.compute_atomic_dipole_h2o(
            traj=traj,
            wannier=wannier,
            type_O=1,
            type_H=2,
            r_bond=1.3,
            a0=a0,
            save_datas=[output_dir / name_h2o, output_dir / name_adw],
        )

        np.testing.assert_array_equal(np.load(sample_dir / name_h2o), np.load(output_dir / name_h2o))
        np.testing.assert_array_equal(np.load(sample_dir / name_adw), np.load(output_dir / name_adw))

    def test_compute_surface_ir_spectra_h2o(self):
        # corresponds to file cal_sur_ir.py
        dt = 0.0005
        window = 50000

        h2o = np.load(sample_dir / "h2o.npy")
        atomic_dipole = np.load(sample_dir / "atomic_dipole_wan.npy")
        atomic_dipole = atomic_dipole.reshape(atomic_dipole.shape[0], -1, 3)
        name = "ir_sp.dat"

        md_spectra.compute_surface_ir_spectra_h2o(
            h2o=h2o,
            atomic_dipole=atomic_dipole,
            dt=dt,
            window=window,
            z1_min=16.0,
            z1_max=17.4,
            z2_min=20.0,
            z2_max=25.0,
            z3_min=27.6,
            z3_max=29.0,
            z_total_min=16.0,
            z_total_max=29.0,
            z_bin=0.4,
            width=25,
            temperature=330.0,
            save_plot=output_dir / "ir_sp.png",
            save_data=output_dir / name,
        )

        # TODO: Confirm why data accuracy is lost
        np.testing.assert_allclose(np.loadtxt(sample_dir / name), np.loadtxt(output_dir / name), atol=1e-4)

    def test_compute_surface_sfg_h2o(self):
        dt = 0.0005
        window = 50000

        h2o = np.load(sample_dir / "h2o.npy")
        atomic_dipole = np.load(sample_dir / "atomic_dipole_wan.npy").reshape(h2o.shape[0], -1, 3)
        atomic_polar = np.load(sample_dir / "atomic_polar_wan.npy").reshape(h2o.shape[0], -1, 3, 3)
        name = "SFG.dat"

        md_spectra.compute_surface_sfg_h2o(
            h2o=h2o,
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
            save_plot=output_dir / "sfg.png",
            save_data=output_dir / name,
        )

        np.testing.assert_allclose(np.loadtxt(sample_dir / name), np.loadtxt(output_dir / name), atol=1e-4)

    def test_compute_bulk_ir_h2o(self):
        dt = 0.0005
        window = 2000

        h2o = np.load(sample_dir / "traj/set.000/h2o.npy")
        cells = np.load(sample_dir / "traj/set.000/box.npy").reshape(h2o.shape[0], 3, 3)
        atomic_dipole = np.load(sample_dir / "traj/set.000/atomic_dipole_wan_h2o.npy").reshape(h2o.shape[0], -1, 3)
        name = "ir_spectrum.npy"

        md_spectra.compute_bulk_ir_h2o(
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
            M=20000,
            filter_type="lorenz",
            save_plot=output_dir / "bulk_ir.png",
            save_data=output_dir / name,
        )

        np.testing.assert_allclose(np.load(sample_dir / name), np.load(output_dir / name), atol=1e-4)

    def test_compute_bulk_raman_h2o(self):
        dt = 0.0005
        window = 2000

        h2o = np.load(sample_dir / "traj/set.000/h2o.npy")
        cells = np.load(sample_dir / "traj/set.000/box.npy").reshape(h2o.shape[0], 3, 3)
        atomic_polar = np.load(sample_dir / "traj/set.000/atomic_polar_wan_h2o.npy").reshape(h2o.shape[0], -1, 3, 3)
        name = "br.npy"

        md_spectra.compute_bulk_raman_h2o(
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
            M=20000,
            filter_type="lorenz",
            save_plots=[
                output_dir / "bulk_raman_iso.png",
                output_dir / "bulk_raman_aniso.png",
                output_dir / "bulk_raman_aniso_low.png",
            ],
            save_data=output_dir / name,
        )

        np.testing.assert_allclose(np.load(sample_dir / name), np.load(output_dir / name), atol=1e-4)

    def test_compute_surface_raman_h2o(self):
        dt = 0.0005
        window = 2000

        h2o = np.load(sample_dir / "traj/set.000/h2o.npy")
        oh = np.load(sample_dir / "traj/set.000/oh.npy")
        cells = np.load(sample_dir / "traj/set.000/box.npy").reshape(h2o.shape[0], 3, 3)
        atomic_polar_h2o = np.load(sample_dir / "traj/set.000/atomic_polar_wan_h2o.npy").reshape(h2o.shape[0], -1, 3, 3)
        atomic_polar_oh = np.load(sample_dir / "traj/set.000/atomic_polar_wan_oh.npy").reshape(h2o.shape[0], -1, 3, 3)
        name = "sr.npy"

        md_spectra.compute_surface_raman_h2o(
            h2o=h2o,
            oh=oh,
            cells=cells,
            atomic_polar_h2o=atomic_polar_h2o,
            atomic_polar_oh=atomic_polar_oh,
            dt=dt,
            window=window,
            z0=21.5,
            zc=10.0,
            zw=0.5,
            rc=6.0,
            width=240,
            temperature=330.0,
            M=20000,
            filter_type="lorenz",
            save_plots=[
                output_dir / "sur_raman_iso.png",
                output_dir / "sur_raman_aniso.png",
                output_dir / "sur_raman_aniso_low.png",
            ],
            save_data=output_dir / name,
        )

        np.testing.assert_allclose(np.load(sample_dir / name), np.load(output_dir / name), atol=1e-4)


if __name__ == "__main__":
    # unittest.main()

    def test_specific(test_name: str):
        suite = unittest.TestSuite()
        suite.addTest(TestMdSpectra(test_name))
        runner = unittest.TextTestRunner()
        runner.run(suite)

    test_specific(
        "test_compute_surface_raman_h2o",
    )
