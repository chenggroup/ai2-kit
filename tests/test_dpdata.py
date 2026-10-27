from pathlib import Path

import numpy as np
import dpdata

from ai2_kit.tool.dpdata import DpdataTool
from tests.helpers import run_cli


def test_dpdata_tool_roundtrips_labeled_cp2k_output(tmp_path):
    sample = Path(__file__).parent / "data/shared/h2o.cp2k.output"
    original = dpdata.LabeledSystem(str(sample), fmt="cp2k/output")
    output = tmp_path / "deepmd"

    DpdataTool(systems=[original]).write(str(output))
    roundtrip = DpdataTool().read(str(output / "*"), fmt="deepmd/npy")._systems[0]

    assert roundtrip.get_nframes() == original.get_nframes()
    np.testing.assert_array_equal(roundtrip.data["atom_names"], original.data["atom_names"])
    np.testing.assert_array_equal(roundtrip.data["atom_types"], original.data["atom_types"])
    for field in ("coords", "cells", "energies", "forces"):
        np.testing.assert_allclose(roundtrip.data[field], original.data[field])


def test_dpdata_cli_roundtrips_labeled_cp2k_output(tmp_path):
    sample = Path(__file__).parent / "data/shared/h2o.cp2k.output"
    output = tmp_path / "deepmd"
    run_cli(
        "tool",
        "dpdata",
        "read",
        str(sample),
        "--fmt",
        "cp2k/output",
        "-",
        "write",
        str(output),
    )

    original = dpdata.LabeledSystem(str(sample), fmt="cp2k/output")
    systems = DpdataTool().read(str(output / "*"), fmt="deepmd/npy")._systems
    assert len(systems) == 1
    roundtrip = systems[0]
    assert roundtrip.get_nframes() == original.get_nframes()
    np.testing.assert_array_equal(roundtrip.data["atom_names"], original.data["atom_names"])
    np.testing.assert_array_equal(roundtrip.data["atom_types"], original.data["atom_types"])
    for field in ("coords", "cells", "energies", "forces"):
        np.testing.assert_allclose(roundtrip.data[field], original.data[field])
