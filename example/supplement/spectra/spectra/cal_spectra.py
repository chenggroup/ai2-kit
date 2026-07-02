# Spectra Calculator
import numpy as np
from ai2_kit.feat.spectrum import md_spectra

dt = 0.004
window = 500
h2o = np.load("../run/data/set.000/h2o.npy")
cells = np.load("../run/data/set.000/box.npy").reshape(h2o.shape[0], 3, 3)
atomic_dipole = np.load("../run/data/set.000/atomic_dipole_wan.npy").reshape(h2o.shape[0], -1, 3)
atomic_polar = np.load("../run/data/set.000/atomic_polar_wan.npy").reshape(h2o.shape[0], -1, 3, 3)

md_spectra.compute_bulk_ir_h2o(
    h2o=h2o,
    cells=cells,
    atomic_dipole=atomic_dipole,
    dt=dt,
    window=window,
    z0=25.0,
    zc=3.0,
    zw=1.0,
    rc=6.0,
    width=240,
    temperature=330.0,
    M=1000,
    filter_type="lorenz",
    nuclear_quantum_factor=0.96,
    save_plot="./bulk_ir.png",
    save_data="./bulk_ir.dat",
)

md_spectra.compute_bulk_raman_h2o(
    h2o=h2o,
    cells=cells,
    atomic_polar=atomic_polar,
    dt=dt,
    window=window,
    z0=25.0,
    zc=3.0,
    zw=1.0,
    rc=6.0,
    width=240,
    temperature=330.0,
    M=1000,
    filter_type="lorenz",
    nuclear_quantum_factor=0.96,
    save_plots=[
        "./bulk_raman_iso.png",
        "./bulk_raman_aniso.png",
        "./bulk_raman_aniso_low.png",
    ],
    save_data="./bulk_raman.dat",
)

md_spectra.compute_surface_sfg_h2o(
    h2o=h2o,
    cells=cells,
    atomic_dipole=atomic_dipole,
    atomic_polar=atomic_polar,
    dt=0.004,
    window=500,
    z0=25.0,
    zc=5.0,
    zw=1.0,
    rc=3.0,
    width=200,
    temperature=330.0,
    M=1000,
    filter_type="lorenz",
    nuclear_quantum_factor=0.96,
    save_plot="./sfg.png",
    save_data="./sfg.dat",
)
