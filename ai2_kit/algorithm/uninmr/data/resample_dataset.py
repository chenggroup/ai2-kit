# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from collections import defaultdict
from unicore.data import BaseWrapperDataset, data_utils
from functools import lru_cache

class EpochResampleDataset(BaseWrapperDataset):
    def __init__(self, dataset, count_dict, seed, max_samples_per_class=None, min_samples_per_class=None):
        super().__init__(dataset)
        self.dataset = dataset
        self.count_dict = count_dict
        self.seed = seed
        self.max_samples_per_class = max_samples_per_class
        self.min_samples_per_class = min_samples_per_class
        self.filtered_indices = self.get_resample(1)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.get_resample(epoch)

    def get_resample(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            # count_dict = defaultdict(int)
            # for itr_class in self.sample_class:
            #     count_dict[itr_class] += 1
            filtered_indices = []
            for itr_class in self.count_dict.keys():
                # indices = [i for i, x in enumerate(self.sample_class) if x == itr_class]
                indices = self.count_dict[itr_class]
                if self.max_samples_per_class is not None:
                    if len(indices) > self.max_samples_per_class:
                        random_indices = np.random.choice(indices, self.max_samples_per_class, replace=False).tolist()
                        indices = random_indices
                if self.min_samples_per_class is not None:
                    if len(indices) < self.min_samples_per_class:
                        random_indices = np.random.choice(indices, self.min_samples_per_class).tolist()
                        indices = random_indices
                filtered_indices.extend(indices)
                # print("epoch", epoch, "class", itr_class, len(indices))
            self.filtered_indices = sorted(filtered_indices)
            # print("dataset", len(self.dataset), "self.filtered_indices", len(self.filtered_indices), self.filtered_indices[:10])
        
            return self.filtered_indices
        
    
    def __len__(self):
        return len(self.filtered_indices)

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        return self.dataset[self.filtered_indices[index]]

class EpochLogResampleDataset(BaseWrapperDataset):
    def __init__(self, dataset, count_dict, seed):
        super().__init__(dataset)
        self.dataset = dataset
        self.count_dict = count_dict
        self.seed = seed
        self.filtered_indices = self.get_resample(1)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.get_resample(epoch)

    def get_resample(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            # Calculate the log values and total log sum for each class
            log_counts = {cls: np.log(len(indices) + 1) for cls, indices in self.count_dict.items()}
            total_log_sum = sum(log_counts.values())

            # Calculate the resampling probability for each class
            resample_probs = {cls: log_counts[cls] / total_log_sum for cls in self.count_dict.keys()}
            
            # Calculate the total number of samples in the original dataset
            total_samples = sum(len(indices) for indices in self.count_dict.values())

            filtered_indices = []
            for itr_class, prob in resample_probs.items():
                indices = self.count_dict[itr_class]
                
                # Sample the number of samples based on log probability
                num_samples = int((total_log_sum) * prob)

                if len(indices) > num_samples:
                    random_indices = np.random.choice(indices, num_samples, replace=False).tolist()
                else:
                    random_indices = np.random.choice(indices, num_samples).tolist()

                filtered_indices.extend(random_indices)

            self.filtered_indices = sorted(filtered_indices)

            return self.filtered_indices
        
    def __len__(self):
        return len(self.filtered_indices)

    @lru_cache(maxsize=16)
    def __getitem__(self, index: int):
        return self.dataset[self.filtered_indices[index]]