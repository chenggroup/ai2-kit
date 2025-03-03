# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import joblib
import os
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
    FunctionTransformer,
)

SCALER_MODE = {
    'minmax': MinMaxScaler(),
    'standard': StandardScaler(),
    'robust': RobustScaler(),
    'maxabs': MaxAbsScaler(),
    'quantile': QuantileTransformer(),
    'power_box': PowerTransformer(method='box-cox'),
    'power_yeo': PowerTransformer(method='yeo-johnson'),
    'normalizer': Normalizer(),
    'log1p': FunctionTransformer(np.log1p),
}

class TargetScaler(object):
    def __init__(self, load_dir=None, scaler_file = 'target_scaler.ss'):
        if load_dir and os.path.exists(os.path.join(load_dir, scaler_file)):
            self.scaler = joblib.load(os.path.join(load_dir, scaler_file))
        else:
            self.scaler = None

    def fit(self, target, num_classes, dump_dir):
        if os.path.exists(os.path.join(dump_dir, 'target_scaler.ss')):
            self.scaler = joblib.load(os.path.join(dump_dir, 'target_scaler.ss'))
        else:
            self.scaler = SCALER_MODE['standard']
            target_selected = target.reshape(-1, num_classes)
            self.scaler.fit(target_selected)

            joblib.dump(self.scaler, os.path.join(dump_dir, 'target_scaler.ss'))

    def transform(self, target):
        return self.scaler.transform(target)


    def inverse_transform(self, target):
        return self.scaler.inverse_transform(target)

    # def fit(self, target, num_classes, dump_dir, mask=None):
    #     if os.path.exists(os.path.join(dump_dir, 'target_scaler.ss')):
    #         return
    #     else:
    #         self.scaler = SCALER_MODE['standard']
    #         if mask == None:
    #             mask = np.ones_like(target)
    #         target_selected = target[mask==1].reshape(-1, num_classes)
    #         self.scaler.fit(target_selected)

    # def transform(self, target, num_classes, mask=None):
    #     if mask == None:
    #         mask = torch.ones_like(target)
    #     return self.scaler.transform(target[mask==1].reshape(-1, num_classes)).reshape(target.shape)


    # def inverse_transform(self, target, num_classes, mask=None):
    #     if mask == None:
    #         mask = torch.ones_like(target)
    #     return self.scaler.inverse_transform(target[mask==1].reshape(-1, num_classes)).reshape(target.shape)




