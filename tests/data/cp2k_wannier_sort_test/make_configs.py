import numpy as np
from ase import io, build


atoms = io.read("coord.xyz")
atoms = build.sort(atoms)
io.write("sorted/coord.xyz", atoms)
random_ids = np.arange(len(atoms))
np.random.shuffle(random_ids)
np.savetxt("random_ids.txt", random_ids, fmt="%d")
io.write("random/coord.xyz", atoms[random_ids])

coord_ref = atoms[random_ids].get_positions()
coord_test = io.read("random/coord.xyz").get_positions()
np.testing.assert_allclose(coord_ref, coord_test)
