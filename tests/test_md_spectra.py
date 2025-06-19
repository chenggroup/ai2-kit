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

data_dir = Path(__file__).parent / "data-sample"


class TestMdSpectra(unittest.TestCase):
    def _is_files_identical(self, file1: str, file2: str, block_size: int = 655436):
        try:
            size1 = os.path.getsize(file1)
            size2 = os.path.getsize(file2)
            if size1 != size2:
                return False

            with open(file1, "rb") as f1, open(file2, "rb") as f2:
                while True:
                    block1 = f1.read(block_size)
                    block2 = f2.read(block_size)
                    if not block1 and not block2:
                        return True
                    if block1 != block2:
                        return False

        except OSError as e:
            print(f"File Error: {e}")
            return False

    def test_extract_atomic_polar_from_traj_h2o(self):
        # corresponds to file cal_polar_wan.py
        traj = dpdata.System(data_dir / "md_spectra_input/traj", fmt="deepmd/npy")
        polar: np.ndarray = np.load(data_dir / "md_spectra_input/wannier_polar.npy")
        polar = -polar.reshape(polar.shape[0], -1, 3, 3)

        atomic_polar = md_spectra.extract_atomic_polar_from_traj_h2o(
            traj=traj,
            polar=polar,
            type_O=0,
            type_H=1,
            r_bond=1.3,
            save_data=data_dir / "md_spectra_output" / "atomic_polar_wan_out.npy",
        )
        assert self._is_files_identical(
            data_dir / "md_spectra_output" / "atomic_polar_wan_out.npy",
            data_dir / "md_spectra_output" / "atomic_polar_wan.npy",
        )


if __name__ == "__main__":
    unittest.main()
