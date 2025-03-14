# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unicore.data import BaseWrapperDataset
from functools import lru_cache
import torch

def copy_tensor(src, dst):
    assert dst.numel() == src.numel()
    dst.copy_(src)

def collate_tokens(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

def collate_tokens_2d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of (batchsize, n, n) tensors into a padded (batchsize, size=max(n), size=max(n)) tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, size).fill_(pad_idx)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):, size - len(v):] if left_pad else res[i][:len(v), :len(v)])
    return res

def collate_tokens_nm(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of (batchsize, n, m) tensors into a padded (batchsize, size=max(n), m) tensor."""
    size = max(v.size(0) for v in values)
    m = values[0].size(1)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, m).fill_(pad_idx)
    
    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :,:] if left_pad else res[i][: len(v),:])
    return res

class PadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return collate_tokens(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)


class LeftPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=True)


class RightPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=False)


class RightPadDataset2D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx,left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
    def collater(self, samples):
        return collate_tokens_2d(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
    
class RightPadDataset2D0(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx,left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
    def collater(self, samples):
        return collate_tokens_nm(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)

class PrependAndAppend2DDataset(BaseWrapperDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            h, w = item.size(-2), item.size(-1)
            new_item = torch.full((h+2, w+2), self.token).type_as(item)
            new_item[1:-1,1:-1] = item
            return new_item
        return item

class PrependAndAppend3DDataset(BaseWrapperDataset):
    def __init__(self, dataset, token=None):
        super().__init__(dataset)
        self.token = token

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.token is not None:
            h, w, d = item.size(-3), item.size(-2), item.size(-1)
            new_item = torch.full((h+2, w+2, d), self.token).type_as(item)
            new_item[1:-1,1:-1,:] = item
            return new_item
        return item
    
def collate_tokens_3d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 3d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values)
    d = values[0].size(-1)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, size, d).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):, size - len(v):, :] if left_pad else res[i][:len(v), :len(v), :])
    return res

class RightPadDataset3D(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx,left_pad=False):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
    def collater(self, samples):
        return collate_tokens_3d(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)