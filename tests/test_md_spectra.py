from pathlib import Path
import unittest
import os

import numpy as np
import dpdata

data_dir = Path(__file__).parent / "data-sample"


class TestMdSpectra(unittest.TestCase):
    def _is_files_identical(file1: str, file2: str):
        SIZE_CAP = 4096
        try:
            size1 = os.path.getsize(file1)
            size2 = os.path.getsize(file2)
            if size1 != size2:
                return False
            if size1 > SIZE_CAP:
                raise ValueError(f"File exceeds size limit({SIZE_CAP}bytes)")

            with open(file1, "rb") as f1, open(file2, "rb") as f2:
                content1 = f1.read()
                content2 = f2.read()
                return content1 == content2

        except (OSError, ValueError) as e:
            print(f"File Error: {e}")
            return False

    def test_extract_atomic_polar_from_traj(self):
        # corresponds to file cal_polar_wan.py
        pass


if __name__ == "__main__":
    unittest.main()
