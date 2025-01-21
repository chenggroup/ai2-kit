# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from unicore.data import BaseWrapperDataset
import numpy as np

class LatticeNormalizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, abc, angles):
        super().__init__(dataset)
        self.dataset = dataset
        self.abc = abc
        self.angles = angles
    
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        abc = np.array(self.dataset[idx][self.abc])
        angles = np.array(self.dataset[idx][self.angles])
        lattices = normalize(abc, angles)
        return lattices

class LatticeMatrixNormalizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, lattice_matrix):
        super().__init__(dataset)
        self.dataset = dataset
        self.lattice_matrix = lattice_matrix
    
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        lattices = np.array(self.dataset[idx][self.lattice_matrix])
        return lattices

def normalize(abc, angles):
    indices = np.argsort(abc)
    abc = abc[indices]
    angles = angles[indices]
    angles = [min(item, 180.0-item) for item in angles]
    angles = np.array(angles) / 180.0 * np.pi
    lattices = np.concatenate([abc, angles]).astype(np.float32)
    return lattices

def normalize_v2(abc, angles):
    # indices = np.argsort(abc)
    # abc = abc[indices]
    # angles = angles[indices]
    # angles = [min(item, 180.0-item) for item in angles]
    angles = np.array(angles) / 180.0 * np.pi
    # lattices = np.concatenate([abc, angles]).astype(np.float32)
    lattices = angles.astype(np.float32)
    return lattices



