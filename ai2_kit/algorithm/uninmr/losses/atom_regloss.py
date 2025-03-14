# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
from unicore import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from unicore.losses import UnicoreLoss, register_loss

from ..utils import TargetScaler


@register_loss("atom_regloss_mse")
class AtomRegMSELoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        # self.target_scaler = TargetScaler()
        self.target_scaler = task.target_scaler

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"],
                           features_only=True,
                           classification_head_name=self.args.classification_head_name)
        select_atom = sample["net_input"]["select_atom"].view(-1, 1)
        sample_size = (select_atom==1).sum()
        src_token = sample["net_input"]["src_tokens"].view(-1, 1)
        # _mean, _std, _normal_type = ATTR_REGESTRY[self.args.task_name]
        # normalizer = Normalization(_mean, _std, _normal_type)
        loss = self.compute_loss(net_output[0], sample, reduce=reduce)
        if not self.training:
            logging_output = {
                "loss": loss.data,
                "predict": self.target_scaler.inverse_transform(net_output[0].view(-1, self.args.num_classes).data.cpu()).astype('float32'),
                "target": self.target_scaler.inverse_transform((sample["target"]["finetune_target"].view(-1, self.args.num_classes))[select_atom==1].view(-1, self.args.num_classes).data.cpu()).astype('float32'),
                "src_token": src_token,
                "select_atom": select_atom,
                "sample_size": sample_size,
                "matid": sample["matid"],
                "num_task": self.args.num_classes,
                # "encoder_rep": net_output[6],
            }
            # print("predict_output", self.target_scaler.inverse_transform(net_output[0].view(-1, self.args.num_classes).data.cpu()))
            # print("target_out", (sample["target"]["finetune_target"].view(-1, self.args.num_classes))[select_atom==1].view(-1, self.args.num_classes).data.cpu())
            # print("predict", self.target_scaler.inverse_transform(net_output[0].view(-1, self.args.num_classes).data.cpu()).astype('float32'))
            # print("target", self.target_scaler.inverse_transform((sample["target"]["finetune_target"].view(-1, self.args.num_classes))[select_atom==1].view(-1, self.args.num_classes).data.cpu()).astype('float32'))
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "num_task": self.args.num_classes,
            }
        logging_output['bsz'] = sample_size
        return loss, sample_size, logging_output

    def compute_loss(self, net_output, sample, reduce=True):
        select_atom = sample["net_input"]["select_atom"].view(-1, 1)
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = sample['target']['finetune_target'].view(-1, self.args.num_classes).float()
        # normalize_targets = torch.from_numpy(self.target_scaler.transform(targets.cpu())).to(targets.device).float()
        loss = F.mse_loss(    # l1_loss mse_loss
            predicts,
            targets[select_atom==1].view(-1, self.args.num_classes),
            # reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        num_tasks = logging_outputs[0]["num_task"]
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / num_tasks, sample_size, round=3
        )
        def reg_metrics(targets, predicts):
            r2 = r2_score(targets, predicts)
            mae = mean_absolute_error(targets, predicts)
            mse = mean_squared_error(targets, predicts)
            rmse = math.sqrt(mse)
            return r2, mae, mse, rmse

        if 'valid' in split or 'test' in split:
            predicts = np.concatenate([log.get("predict") for log in logging_outputs], axis=0)
            # predicts = predicts.detach().cpu().numpy()
            targets = np.concatenate([log.get("target") for log in logging_outputs], axis=0)
            # targets = targets.detach().cpu().numpy()
            #####
            r2, mae, mse, rmse = reg_metrics(targets, predicts)
            metrics.log_scalar("{}_r2".format(split), r2, sample_size, round=4)
            metrics.log_scalar("{}_mae".format(split), mae, sample_size, round=4)
            metrics.log_scalar("{}_mse".format(split), mse, sample_size, round=4)
            metrics.log_scalar("{}_rmse".format(split), rmse, sample_size, round=4)
            #####
            src_tokens = torch.cat([log.get("src_token")[log.get("select_atom")==1] for log in logging_outputs], dim=0)
            src_tokens = src_tokens.detach().cpu().numpy()
            elemenets = set(src_tokens)
            if len(elemenets) > 1:
                for element in elemenets:
                    element_targets = targets[src_tokens==element]
                    element_predicts = predicts[src_tokens==element]
                    r2, mae, mse, rmse = reg_metrics(element_targets, element_predicts)
                    element_sample_size = len(src_tokens[src_tokens==element])
                    metrics.log_scalar("{}_{}_r2".format(split, [element]), r2, element_sample_size, round=4)
                    metrics.log_scalar("{}_{}_mae".format(split, [element]), mae, element_sample_size, round=4)
                    metrics.log_scalar("{}_{}_mse".format(split, [element]), mse, element_sample_size, round=4)
                    metrics.log_scalar("{}_{}_rmse".format(split, [element]), rmse, element_sample_size, round=4)

@register_loss("atom_regloss_mae")
class AtomRegMAELoss(AtomRegMSELoss):
    def __init__(self, task):
        super().__init__(task)
        # self.target_scaler = TargetScaler()
        self.target_scaler = task.target_scaler
    # def forward(self, model, sample, reduce=True):
    #     """Compute the loss for the given sample.

    #     Returns a tuple with three elements:
    #     1) the loss
    #     2) the sample size, which is used as the denominator for the gradient
    #     3) logging outputs to display while training
    #     """
    #     net_output = model(**sample["net_input"],
    #                        features_only=True,
    #                        classification_head_name=self.args.classification_head_name)
    #     select_atom = sample["net_input"]["select_atom"].view(-1, 1)
    #     sample_size = (select_atom==1).sum()
    #     src_token = sample["net_input"]["src_tokens"].view(-1, 1)
    #     # _mean, _std, _normal_type = ATTR_REGESTRY[self.args.task_name]
    #     # normalizer = Normalization(_mean, _std, _normal_type)
    #     loss = self.compute_loss(net_output[0], sample, reduce=reduce)

    #     if not self.training:
    #         logging_output = {
    #             "loss": loss.data,
    #             "predict": (torch.from_numpy(self.target_scaler.inverse_transform(net_output[0].view(-1, self.args.num_classes).data.cpu())).to(net_output[0].device).float()),
    #             "target": (sample["target"]["finetune_target"].view(-1, self.args.num_classes).data),
    #             "src_token": src_token,
    #             "select_atom": select_atom,
    #             "sample_size": sample_size,
    #             "num_task": self.args.num_classes,
    #         }
    #         # print(logging_output["target"][:100])
    #     else:
    #         logging_output = {
    #             "loss": loss.data,
    #             "sample_size": sample_size,
    #             "num_task": self.args.num_classes,
    #         }
    #     logging_output['bsz'] = sample_size
    #     return loss, sample_size, logging_output

    def compute_loss(self, net_output, sample, reduce=True):
        select_atom = sample["net_input"]["select_atom"].view(-1, 1)
        predicts = net_output.view(-1, self.args.num_classes).float()
        targets = sample['target']['finetune_target'].view(-1, self.args.num_classes).float()
        # print("loss.py", predicts, targets)
        # normalize_targets = torch.from_numpy(self.target_scaler.transform(targets.cpu())).to(targets.device).float()
        loss = F.l1_loss(    # l1_loss mse_loss
            predicts,
            targets[select_atom==1].view(-1, self.args.num_classes),
            # reduction="sum" if reduce else "none",
        )
        # l1_loss = nn.L1Loss()
        # loss = l1_loss(    # l1_loss mse_loss
        #     predicts[select_atom==1],
        #     normalize_targets[select_atom==1],
        # )
        return loss

    # @staticmethod
    # def reduce_metrics(logging_outputs, split='valid') -> None:
    #     """Aggregate logging outputs from data parallel training."""
    #     loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
    #     sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
    #     num_tasks = logging_outputs[0]["num_task"]
    #     # we divide by log(2) to convert the loss from base e to base 2
    #     metrics.log_scalar(
    #         "loss", loss_sum / sample_size / num_tasks, sample_size, round=3
    #     )
    #     def reg_metrics(targets, predicts):
    #         r2 = r2_score(targets, predicts)
    #         mae = mean_absolute_error(targets, predicts)
    #         mse = mean_squared_error(targets, predicts)
    #         rmse = math.sqrt(mse)
    #         return r2, mae, mse, rmse

    #     if 'valid' in split or 'test' in split:
    #         predicts = torch.cat([log.get("predict") for log in logging_outputs], dim=0)
    #         predicts = predicts.detach().cpu().numpy()
    #         targets = torch.cat([log.get("target") for log in logging_outputs], dim=0)
    #         targets = targets.detach().cpu().numpy()
    #         #####
    #         r2, mae, mse, rmse = reg_metrics(targets, predicts)
    #         metrics.log_scalar("{}_r2".format(split), r2, sample_size, round=4)
    #         metrics.log_scalar("{}_mae".format(split), mae, sample_size, round=4)
    #         metrics.log_scalar("{}_mse".format(split), mse, sample_size, round=4)
    #         metrics.log_scalar("{}_rmse".format(split), rmse, sample_size, round=4)
    #         #####
    #         src_tokens = torch.cat([log.get("src_token") for log in logging_outputs], dim=0)
    #         src_tokens = src_tokens.detach().cpu().numpy()
    #         elemenets = set(src_tokens)
    #         if len(elemenets) > 1:
    #             for element in elemenets:
    #                 element_targets = targets[src_tokens==element]
    #                 element_predicts = predicts[src_tokens==element]
    #                 r2, mae, mse, rmse = reg_metrics(element_targets, element_predicts)
    #                 element_sample_size = len(src_tokens[src_tokens==element])
    #                 metrics.log_scalar("{}_{}_r2".format(split, [element]), r2, element_sample_size, round=4)
    #                 metrics.log_scalar("{}_{}_mae".format(split, [element]), mae, element_sample_size, round=4)
    #                 metrics.log_scalar("{}_{}_mse".format(split, [element]), mse, element_sample_size, round=4)
    #                 metrics.log_scalar("{}_{}_rmse".format(split, [element]), rmse, element_sample_size, round=4)


