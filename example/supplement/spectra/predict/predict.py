import numpy as np
from deepmd.infer import DeepDipole, DeepPolar

BATCH_SIZE = 64
coord = np.load('../simulate/data/set.000/coord.npy')
cell = np.load('../simulate/data/set.000/box.npy')
atype = np.loadtxt('../simulate/data/type.raw').astype(int)
nframes = coord.shape[0]
print(nframes)

dpdipole = DeepDipole("../train/dipole/dipole.pb")
batch = 0
dipole_all = []
while batch < nframes:
    print("-----------------------------------------", "current batch", batch, "-----------------------------------------")
    batch_n = min(batch + BATCH_SIZE, nframes)
    dipole = dpdipole.eval(coord[batch:batch_n], cell[batch:batch_n], atype)
    dipole_all.append(dipole)
    batch = batch_n
dipole_vec = np.concatenate(dipole_all, axis = 0).reshape([nframes, -1])
np.save('../simulate/data/set.000/wannier_dipole.npy', dipole_vec.astype(np.float32))

dppolar = DeepPolar("../train/polar/polar.pb")
batch = 0
polar_all = []
while batch < nframes:
    print("-----------------------------------------", "current batch", batch, "-----------------------------------------")
    batch_n = min(batch + BATCH_SIZE, nframes)
    polar = dppolar.eval(coord[batch:batch_n], cell[batch:batch_n], atype)
    polar_all.append(polar)
    batch = batch_n
polar_vec = np.concatenate(polar_all, axis = 0).reshape([nframes, -1])
np.save('../simulate/data/set.000/wannier_polar.npy', polar_vec.astype(np.float32))