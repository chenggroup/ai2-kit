# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss


@register_loss("unimat_rcut")
class UniMatRCutLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()
        self.seed = task.seed

    def forward(self, model, sample, reduce=True):
        input_key = "net_input"
        target_key = "target"
        masked_tokens = sample[target_key]["tokens_target"].ne(self.padding_idx)
        select_atom = sample[input_key]["select_atom"]
        sample_size = (select_atom==1).long().sum()
        logits_encoder, encoder_distance, encoder_coord, lattice, x_norm, delta_encoder_pair_rep_norm = model(**sample[input_key], encoder_masked_tokens=(select_atom==1))
        target = sample[target_key]["tokens_target"]
        if select_atom is not None:
            target = target[select_atom==1]
        masked_token_loss = F.nll_loss(
            F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction="mean",
        )
        masked_pred = logits_encoder.argmax(dim=-1)
        masked_hit = (masked_pred == target).long().sum()
        masked_cnt = sample_size
        loss = masked_token_loss * self.args.masked_token_loss
        if not self.training:
            logging_output = {
                "sample_size": 1,
                "bsz": sample[target_key]["tokens_target"].size(0),
                "seq_len": sample[target_key]["tokens_target"].size(1) * sample[target_key]["tokens_target"].size(0),
                "masked_token_loss": masked_token_loss.data,
                "masked_token_hit": masked_hit.data,
                "masked_token_cnt": masked_cnt,
            }
        else:
            logging_output = {
                "sample_size": 1,
                "bsz": sample[target_key]["tokens_target"].size(0),
                "seq_len": sample[target_key]["tokens_target"].size(1) * sample[target_key]["tokens_target"].size(0),
            }
        if encoder_distance is not None:
            dist_masked_tokens = (select_atom==1)
            masked_distance = encoder_distance[dist_masked_tokens, :]
            if self.args.global_distance:
                masked_distance_target = sample[target_key]["distance_target"][dist_masked_tokens][:,:,:4]
            else:
                masked_distance_target = sample[target_key]["distance_target"][dist_masked_tokens, :]
            non_pad_pos = masked_distance_target > 0
            non_pad_pos &= masked_distance_target < self.args.dist_threshold
            masked_dist_loss = F.smooth_l1_loss(
                masked_distance[non_pad_pos].view(-1).float(),
                masked_distance_target[non_pad_pos].view(-1),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + masked_dist_loss * self.args.masked_dist_loss
            if not self.training:
                logging_output["masked_dist_loss"] = masked_dist_loss.data

        if encoder_coord is not None:
            coord_target = sample[target_key]["coord_target"]
            masked_coord_loss = F.smooth_l1_loss(
                encoder_coord[select_atom==1].view(-1, 3).float(),
                coord_target[select_atom==1].view(-1, 3),
                reduction="mean",
                beta=1.0,
            )
            loss = loss + masked_coord_loss * self.args.masked_coord_loss
            if not self.training:
                logging_output["masked_coord_loss"] = masked_coord_loss.data

        if lattice is not None:
            lattice_target = sample[target_key]["lattice_target"]
            # select_indices = select_atom.nonzero(as_tuple=True)
            lattice_loss = F.mse_loss(
                lattice.float(),
                lattice_target.unsqueeze(1).repeat(1, select_atom.shape[1], 1)[select_atom==1],
                reduction="mean",
            )
            loss = loss + lattice_loss * self.args.lattice_loss
            if not self.training:
                logging_output["lattice_loss"] = lattice_loss.data

        if x_norm is not None:
            loss = loss + self.args.x_norm_loss * x_norm
            if not self.training:
                logging_output["x_norm_loss"] = x_norm.data

        if delta_encoder_pair_rep_norm is not None:
            loss = loss + self.args.delta_pair_repr_norm_loss * delta_encoder_pair_rep_norm
            if not self.training:
                logging_output["delta_pair_repr_norm_loss"] = delta_encoder_pair_rep_norm.data

        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar(
            "seq_len", seq_len / bsz, 1, round=3
        )
        if 'valid' in split or 'test' in split:
            masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
            metrics.log_scalar("masked_token_loss", masked_loss / sample_size, sample_size, round=3)

            masked_acc = sum(log.get("masked_token_hit", 0) for log in logging_outputs) / sum(log.get("masked_token_cnt", 0) for log in logging_outputs)
            metrics.log_scalar("masked_acc", masked_acc, sample_size, round=3)

            masked_coord_loss = sum(log.get("masked_coord_loss", 0) for log in logging_outputs)
            if masked_coord_loss > 0:
                metrics.log_scalar("masked_coord_loss", masked_coord_loss / sample_size, sample_size, round=3)

            masked_dist_loss = sum(log.get("masked_dist_loss", 0) for log in logging_outputs)
            if masked_dist_loss > 0:
                metrics.log_scalar("masked_dist_loss", masked_dist_loss / sample_size, sample_size, round=3)
            
            lattice_Loss = sum(log.get("lattice_loss", 0) for log in logging_outputs)
            if lattice_Loss > 0:
                metrics.log_scalar("lattice_loss", lattice_Loss / sample_size, sample_size, round=3)

            x_norm_loss = sum(log.get("x_norm_loss", 0) for log in logging_outputs)
            if x_norm_loss > 0:
                metrics.log_scalar("x_norm_loss", x_norm_loss / sample_size, sample_size, round=3)

            delta_pair_repr_norm_loss = sum(log.get("delta_pair_repr_norm_loss", 0) for log in logging_outputs)
            if delta_pair_repr_norm_loss > 0:
                metrics.log_scalar("delta_pair_repr_norm_loss", delta_pair_repr_norm_loss / sample_size, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
