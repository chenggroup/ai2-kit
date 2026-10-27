from pathlib import Path

import numpy as np

from ai2_kit.tool.ase import AseTool
from tests.helpers import run_cli


def test_write_each_frame_delegates_to_write_frames():
    tool = AseTool.__new__(AseTool)

    assert tool.write_each_frame == tool.write_frames


def test_ase_tool_reads_writes_and_reads_back_xyz(tmp_path):
    sample = Path(__file__).parent / "data/shared/h2o.xyz"
    output = tmp_path / "roundtrip.xyz"

    atoms = AseTool().read(str(sample), index=0)
    atoms.write(str(output), format="extxyz")
    roundtrip = AseTool().read(str(output), format="extxyz")

    assert roundtrip.size() == 1
    np.testing.assert_array_equal(
        roundtrip._atoms_arr[0].get_chemical_symbols(),
        atoms._atoms_arr[0].get_chemical_symbols(),
    )
    np.testing.assert_allclose(
        roundtrip._atoms_arr[0].get_positions(), atoms._atoms_arr[0].get_positions()
    )
    np.testing.assert_allclose(
        roundtrip._atoms_arr[0].get_cell(), atoms._atoms_arr[0].get_cell()
    )


def test_ase_cli_reads_writes_and_reads_back_xyz(tmp_path):
    sample = Path(__file__).parent / "data/shared/h2o.xyz"
    output = tmp_path / "roundtrip.xyz"
    run_cli(
        "tool",
        "ase",
        "read",
        str(sample),
        "--index",
        "0",
        "-",
        "write",
        str(output),
        "--format",
        "extxyz",
    )

    atoms = AseTool().read(str(sample), index=0)
    roundtrip = AseTool().read(str(output), format="extxyz")
    assert roundtrip.size() == 1
    np.testing.assert_array_equal(
        roundtrip._atoms_arr[0].get_chemical_symbols(),
        atoms._atoms_arr[0].get_chemical_symbols(),
    )
    np.testing.assert_allclose(
        roundtrip._atoms_arr[0].get_positions(), atoms._atoms_arr[0].get_positions()
    )
    np.testing.assert_allclose(
        roundtrip._atoms_arr[0].get_cell(), atoms._atoms_arr[0].get_cell()
    )
