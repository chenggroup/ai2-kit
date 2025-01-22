# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class NormalizeDataset(BaseWrapperDataset):
    def __init__(self, dataset, coordinates):
        self.dataset = dataset
        self.coordinates = coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        dd = self.dataset[index].copy()
        coordinates = dd[self.coordinates]
        coordinates = coordinates - coordinates.mean(axis=0)
        dd[self.coordinates] = coordinates.astype(np.float32)
        return dd

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)

class TargetScalerDataset(BaseWrapperDataset):
    def __init__(self, dataset, target_scaler, num_classes):
        self.dataset = dataset
        self.target_scaler = target_scaler
        self.num_classes = num_classes

    def __getitem__(self, index: int):
        return self.target_scaler.transform(self.dataset[index].reshape(-1, self.num_classes)).reshape(-1)

