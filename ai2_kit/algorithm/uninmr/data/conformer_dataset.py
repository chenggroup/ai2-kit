# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset, data_utils


class ConformerSampleDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        ret = self.dataset[index]
        size = len(ret[self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        ret["coordinates"] = ret[self.coordinates][sample_idx]
        return ret

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class TTADataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, conf_size=10):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.conf_size = conf_size
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        mat_idx = index // self.conf_size
        ret = self.dataset[mat_idx]
        coord_idx = index % self.conf_size
        ret["coordinates"] =  np.array(ret[self.coordinates][coord_idx])

        return ret

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

class TTAIndexDataset(BaseWrapperDataset):
    def __init__(self, dataset, conf_size=10):
        self.dataset = dataset
        self.conf_size = conf_size

    def __len__(self):
        return len(self.dataset) * self.conf_size

    def __getitem__(self, index: int):
        mat_idx = index // self.conf_size
        return self.dataset[mat_idx]
