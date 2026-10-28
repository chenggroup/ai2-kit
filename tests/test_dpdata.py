from pathlib import Path

import numpy as np
import dpdata
import pytest

from ai2_kit.tool.dpdata import DpdataTool
from tests.helpers import run_cli


@pytest.fixture(params=[
    pytest.param(("shared/h2o.cp2k.output", "cp2k/output"), id="dpdata"),
    pytest.param(("cp2k_nacl_finite_field/output", "cp2kdata/e_f"), id="cp2kdata"),
])
def cp2k_input(request):
    sample, fmt = request.param
    if fmt == "cp2kdata/e_f":
        pytest.importorskip("cp2kdata")
    return Path(__file__).parent / "data" / sample, fmt


def test_dpdata_tool_roundtrips_labeled_cp2k_output(tmp_path, cp2k_input):
    sample, fmt = cp2k_input
    original = dpdata.LabeledSystem(str(sample), fmt="cp2k/output")
    output = tmp_path / "deepmd"

    DpdataTool().read(str(sample), fmt=fmt).write(str(output))
    systems = DpdataTool().read(str(output / "*"), fmt="deepmd/npy")._systems
    assert len(systems) == 1
    roundtrip = systems[0]

    assert roundtrip.get_nframes() == original.get_nframes()
    np.testing.assert_array_equal(roundtrip.data["atom_names"], original.data["atom_names"])
    np.testing.assert_array_equal(roundtrip.data["atom_types"], original.data["atom_types"])
    for field in ("coords", "cells", "energies", "forces"):
        np.testing.assert_allclose(roundtrip.data[field], original.data[field])


def test_dpdata_cli_roundtrips_labeled_cp2k_output(tmp_path, cp2k_input):
    sample, fmt = cp2k_input
    output = tmp_path / "deepmd"
    run_cli(
        "tool",
        "dpdata",
        "read",
        str(sample),
        "--fmt",
        fmt,
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
