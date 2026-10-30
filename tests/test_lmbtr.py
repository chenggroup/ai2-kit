import numpy as np
from ase.build import molecule
from ase.io import write

from ai2_kit.feat.spectrum.lmbtr.xyz_utils import generate_and_save_lmbtr_descriptors


def test_lmbtr_descriptors_are_invariant_under_rotation_and_translation(tmp_path):
    atoms = molecule("H2O")
    transformed = atoms.copy()
    transformed.rotate(37, "x")
    transformed.translate([1.0, -2.0, 3.0])
    write(tmp_path / "water.xyz", [atoms, transformed])

    descriptors = generate_and_save_lmbtr_descriptors(
        xyz_folder_path=str(tmp_path),
        file_pattern="water.xyz",
        central_atoms=["O"],
        output_path=str(tmp_path / "descriptors.pkl"),
        species=["H", "O"],
        periodic=False,
    )

    assert len(descriptors) == 2
    for descriptor in descriptors:
        assert descriptor.ndim == 1
        assert descriptor.size > 0
        assert np.isfinite(descriptor).all()
        # The concatenated distance and angle descriptors each have unit norm.
        np.testing.assert_allclose(np.linalg.norm(descriptor), np.sqrt(2))
    np.testing.assert_allclose(descriptors[0], descriptors[1], atol=1e-8)
