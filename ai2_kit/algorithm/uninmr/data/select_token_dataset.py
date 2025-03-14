# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class SelectTokenDataset(BaseWrapperDataset):
    def __init__(
        self,
        token_dataset,
        token_mask_dataset=None,
        selected_token=np.array([-1]),
        random_choice=0,
    ):
        self.dataset = token_dataset
        self.token_mask_dataset = token_mask_dataset
        self.selected_token = selected_token
        self.random_choice = random_choice

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        token = self.dataset[index]
        if self.selected_token == np.array([-1]):
            ret = torch.ones_like(token)
        else:
            ret = torch.zeros_like(token)
            for val in self.selected_token:
                ret[token == val] = 1

        if self.token_mask_dataset is not None:
            token_mask = torch.from_numpy(self.token_mask_dataset[index]).long()
            ret = ret & token_mask
        if self.random_choice > 0:
            mask = torch.rand(ret.shape) < self.random_choice
            ret[mask == False] = 0
            ret[0] = 1

        return ret

# class SelectedDataset(BaseWrapperDataset):
#     def __init__(
#         self,
#         select_atom_dataset,
#     ):
#         self.dataset = select_atom_dataset

#     @lru_cache(maxsize=16)
#     def __getitem__(self, index: int):
#         return int(not torch.all(self.dataset[index] == 0))

class FilterDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        filter_list
    ):
        self.dataset = dataset
        self.index_list = [index for index, value in enumerate(filter_list) if value == 1]

    def __len__(self):
        return len(self.index_list)

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        return self.dataset[self.index_list[index]]

