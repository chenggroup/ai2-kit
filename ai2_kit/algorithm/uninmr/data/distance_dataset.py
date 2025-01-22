# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from itertools import product
import torch
from scipy.spatial import distance_matrix
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class DistanceDataset(BaseWrapperDataset):

    def __init__(self, dataset, p=2):
        super().__init__(dataset)
        self.dataset = dataset
        self.p = p

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        pos = self.dataset[idx].view(-1, 3).numpy()
        dist = distance_matrix(pos, pos, self.p).astype(np.float32)
        return torch.from_numpy(dist)

class GlobalDistanceDataset(BaseWrapperDataset):

    def __init__(self, dataset, lattice_matrix_dataset, p=2):
        super().__init__(dataset)
        self.dataset = dataset
        self.lattice_matrix_dataset = lattice_matrix_dataset
        self.p = p

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        pos = self.dataset[idx].view(-1, 3).numpy()    # (n,3)
        pbc_pos = np.tile(pos, (125, 1, 1))                   # (343,n,3)
        lattice_matrix = self.lattice_matrix_dataset[idx].astype(np.float32)
        ks = [0, 1, -1, 2, -2]
        pbc_matrix = np.array(list(product(ks, repeat=3)))
        pbc_pos += np.dot(pbc_matrix, lattice_matrix).reshape(125,-1,3)
        pbc_pos = pbc_pos.reshape(-1,3)                      # (343*n,3)
        
        # dist = distance_matrix(pos, pos, self.p).astype(np.float32)
        dist_pbc = distance_matrix(pbc_pos, pos, self.p).astype(np.float32)       # (343*n,n)
        dist_pbc = (np.sort(dist_pbc.reshape(125,-1,pos.shape[0]),axis=0))[:4].transpose((1, 2, 0))    # (n,n,16)
        return torch.from_numpy(dist_pbc)

# class GlobalDistanceDataset(BaseWrapperDataset):

#     def __init__(self, dataset, lattice_matrix_dataset, p=2):
#         super().__init__(dataset)
#         self.dataset = dataset
#         self.lattice_matrix_dataset = lattice_matrix_dataset
#         self.p = p

#     @lru_cache(maxsize=16)
#     def __getitem__(self, idx):
#         pos = self.dataset[idx].view(-1, 3).numpy()    # (n,3)
#         pbc_pos = np.tile(pos, (27, 1, 1))                   # (27,n,3)
#         lattice_matrix = self.lattice_matrix_dataset[idx].astype(np.float32)
#         x = np.array([-1, 0, 1])
#         X, Y, Z = np.meshgrid(x, x, x)
#         pbc_matrix = np.stack((Y, X, Z), axis=-1).reshape(-1, 3)
#         pbc_pos += np.dot(pbc_matrix, lattice_matrix).reshape(27,-1,3)
#         pbc_pos = pbc_pos.reshape(-1,3)                      # (27*n,3)
        
#         # dist = distance_matrix(pos, pos, self.p).astype(np.float32)
#         dist_pbc = distance_matrix(pbc_pos, pos, 2).astype(np.float32)       # (27*n,n)
#         dist_pbc = np.min(dist_pbc.reshape(27,-1,pos.shape[0]), axis = 0)    # (n,n)
#         return torch.from_numpy(dist_pbc)

class EdgeTypeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_types: int
    ):
        self.dataset = dataset
        self.num_types = num_types

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        node_input = self.dataset[index].clone()
        offset = node_input.view(-1, 1) * self.num_types + node_input.view(1, -1)
        return offset