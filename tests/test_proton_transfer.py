import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from tests.helpers import run_cli


@pytest.mark.parametrize("trajectory", ["manual", "transfer"])
def test_proton_transfer_cli_smoke(tmp_path, monkeypatch, trajectory):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    sample = Path(__file__).resolve().parents[1] / "doc/res/proton-transfer-test-trajectory.xyz"
    frames = read(sample, index=":")
    cell = [12.740, 13.399, 40.985, 90, 90, 90]
    donors = [240, 255, 246]
    if trajectory == "transfer":
        sample = tmp_path / "transfer.xyz"
        cell = [20, 20, 20, 90, 90, 90]
        donors = [0]
        frames = [
            Atoms("OHO", positions=[[10, 10, 9.8], [x, 10, 10], [12, 10, 10.2]])
            for x in [10.8, 11.2, 11.3]
        ]
        write(sample, frames)
    result_dir = tmp_path / "result"
    cli = ("algorithm", "proton-transfer")

    run_cli(
        *cli, "analyze",
        "--input_traj", str(sample),
        "--out_dir", str(result_dir),
        "--cell", json.dumps(cell),
        "--acceptor_elements", '["O"]',
        "--initial_donors", json.dumps(donors),
        "--core_num", "1",
    )

    for donor in donors:
        records = [json.loads(line) for line in (result_dir / f"{donor}.jsonl").read_text().splitlines()]
        assert len(records) == len(frames)
        positions = np.array([record[0] for record in records])
        assert positions.shape == (len(frames), 3)
        assert np.isfinite(positions).all()
        np.testing.assert_allclose(positions[0], frames[0].positions[donor], atol=1e-6)
        for _, transfers in records:
            for acceptor, proton in transfers:
                assert frames[0][acceptor].symbol == "O"
                assert frames[0][proton].symbol == "H"

    donor = donors[0]
    records = [json.loads(line) for line in (result_dir / f"{donor}.jsonl").read_text().splitlines()]
    if trajectory == "transfer":
        assert records[1][1] == [[2, 1]]
    output_traj = tmp_path / "labeled-traj.xyz"
    run_cli(
        *cli, "visualize",
        "--analysis_result", str(result_dir),
        "--input_traj", str(sample),
        "--output_traj", str(output_traj),
        "--initial_donor", str(donor),
        "--cell", json.dumps(cell),
    )
    labeled_frames = read(output_traj, index=":")
    assert len(labeled_frames) == len(frames)
    current_donor = donor
    for original, labeled, (position, transfers) in zip(frames, labeled_frames, records):
        assert len(labeled) == len(original) + 1
        assert labeled[current_donor].symbol == "N"
        assert labeled[-1].symbol == "F"
        np.testing.assert_allclose(labeled[-1].position, position, atol=1e-6)
        if transfers:
            current_donor = transfers[-1][0]

    # The manual's short trajectory contains no transfers for these donors.
    if trajectory == "manual":
        return

    paths = run_cli(
        *cli, "show-transfer-paths",
        "--analysis_result", str(result_dir),
        "--initial_donor", str(donor),
    )
    assert "transfer_path_index" in paths.stdout
    assert "0(1)->2" in paths.stdout
    proton_infos = [
        json.loads(line)
        for line in (result_dir / f"{donor}_proton_infos.jsonl").read_text().splitlines()
    ]
    assert proton_infos == [[1, 1], [1, len(frames)]]

    type_changes = run_cli(
        *cli, "show-type-change",
        "--analysis_result", str(result_dir),
        "--atom_types", json.dumps({"Bridge_O": [0], "Water_O": [2]}),
        "--donors", json.dumps(donors),
    )
    assert "proton transfer type change" in type_changes.stdout
    assert "Bridge_O<->Water_O" in type_changes.stdout
    assert "0 -> 2" in type_changes.stdout

    run_cli(
        *cli, "calculate-distances",
        "--analysis_result", str(result_dir),
        "--input_traj", str(sample),
        "--upper_index", "[0]",
        "--lower_index", "[2]",
        "--initial_donor", str(donor),
        "--interval", "1",
    )
    distances = np.loadtxt(result_dir / f"{donor}_proton_distance_to_interface.jsonl")
    assert distances.shape == (len(frames),)
    assert np.isfinite(distances).all()
    assert (distances >= 0).all()
    np.testing.assert_allclose(distances, 0.2)

    run_cli(
        *cli, "show-distance-change",
        "--analysis_result", str(result_dir),
        "--initial_donor", str(donor),
    )
    plot = result_dir / f"{donor}_proton_distance_to_interface.png"
    assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
