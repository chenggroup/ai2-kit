# Dipole-Polarizability Extractor
import numpy as np
import dpdata
from ai2_kit.feat.spectrum import md_spectra

traj = dpdata.System("../simulate/data", fmt="deepmd/npy")

md_spectra.compute_atomic_dipole_h2o(
    traj=traj,
    wannier=np.load("../simulate/data/set.000/wannier_dipole.npy").reshape(traj.get_nframes(), -1, 3),
    type_O=0,
    type_H=1,
    r_bond=1.2,
    save_datas=["../simulate/data/set.000/h2o.npy", "../simulate/data/set.000/atomic_dipole_wan.npy"],
)

md_spectra.extract_atomic_polar_from_traj_h2o(
    traj=traj,
    polar=np.load("../simulate/data/set.000/wannier_polar.npy").reshape(traj.get_nframes(), -1, 3, 3),
    type_O=0,
    type_H=1,
    r_bond=1.2,
    save_data="../simulate/data/set.000/atomic_polar_wan.npy",
)
