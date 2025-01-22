# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from typing import Optional
import numpy as np
import torch
from unicore.data import Dictionary, data_utils
from unicore.data import BaseWrapperDataset


class MaskPointsDataset(BaseWrapperDataset):

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        coord_dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        noise_type: str,
        noise: float = 1.0,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        select_atom_dataset: Optional[torch.utils.data.Dataset] = None,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.coord_dataset = coord_dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.noise_type = noise_type
        self.noise = noise
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.select_atom_dataset = select_atom_dataset

        if random_token_prob > 0.0:
            weights = np.ones(len(self.vocab))
            weights[vocab.special_index()] = 0
            self.weights = weights / weights.sum()

        self.epoch = None
        if self.noise_type == 'trunc_normal':
            self.noise_f = lambda num_mask: np.clip(np.random.randn(num_mask, 3) * self.noise, a_min=-self.noise*2.0, a_max=self.noise*2.0)
        elif self.noise_type == 'normal':
            self.noise_f = lambda num_mask: np.random.randn(num_mask, 3) * self.noise
        elif self.noise_type == 'uniform':
            self.noise_f = lambda num_mask: np.random.uniform(low=-self.noise, high=self.noise, size=(num_mask, 3))
        else:
            self.noise_f = lambda num_mask: 0.0

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.coord_dataset.set_epoch(epoch)
        self.dataset.set_epoch(epoch)
        self.epoch = epoch

    # def __getitem__(self, index: int):
    #     return self.__getitem_cached__(self.epoch, index)
    
    def __getitem__(self, index: int):
        if self.epoch is None:
            return self.__getitem_cached__(1, index)  # Use a default value if self.epoch is None
        else:
            return self.__getitem_cached__(int(self.epoch), index)  # Convert self.epoch to an integer


    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        ret = {}
        with data_utils.numpy_seed(self.seed, epoch, index):
            item = self.dataset[index]
            coord = self.coord_dataset[index]
            sz = len(item)
            # don't allow empty sequence
            assert sz > 0
            # decide elements to mask
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz + np.random.rand()
            )
            mask_idc = np.random.choice(sz, num_mask, replace=False)
            mask = np.full(sz, False)
            mask[mask_idc] = True
            if self.select_atom_dataset is not None:
                select_mask = self.select_atom_dataset[index].numpy()
                mask[select_mask==1] = True
            ret['targets'] = np.full(len(mask), self.pad_idx)
            ret['targets'][mask] = item[mask]
            ret['targets'] = torch.from_numpy(ret['targets']).long()
            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx

            num_mask = mask.astype(np.int32).sum()
            new_coord = np.copy(coord)
            new_coord[mask, :] += self.noise_f(num_mask)

            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab),
                        num_rand,
                        p=self.weights,
                    )
            ret['atoms'] = torch.from_numpy(new_item).long()
            ret['coordinates'] = torch.from_numpy(new_coord).float()
            return ret