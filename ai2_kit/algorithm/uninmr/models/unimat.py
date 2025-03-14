# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from typing import Dict, Any, List


logger = logging.getLogger(__name__)


@register_model("unimat")
class UniMatModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout", type=float, metavar="D", help="dropout probability for embeddings"
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--lattice-loss",
            type=float,
            metavar="D",
            help="lattice loss ratio",
        )
        parser.add_argument(
            "--gaussian-kernel",
            action="store_true",
            help="use gaussian kernel for distance map",
        )
        # parser.add_argument(
        #     "--atom-descriptor",
        #     action="store_true",
        #     help="use extra atom descriptor",
        # )  
        # parser.add_argument(
        #     "--global-distance",
        #     action="store_true",
        #     help="use global distance",
        # ) 

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(len(dictionary), args.encoder_embed_dim, self.padding_idx)
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=self.embed_tokens.weight,
            )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(K, args.encoder_attention_heads, args.activation_fn)
        if 'gaussian_kernel' in args and args.gaussian_kernel:
            if args.global_distance:
                self.gbf = GlobalGaussianLayer(K, n_edge_type)
            else:
                self.gbf = GaussianLayer(K, n_edge_type)
        else:
            self.gbf = NumericalEmbed(K, n_edge_type)
        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(args.encoder_attention_heads, 1, args.activation_fn)
        if args.masked_dist_loss > 0:
            if args.global_distance:
                self.glodist_head = GlobalDistanceHead(args.encoder_attention_heads, args.activation_fn)
            else:
                self.dist_head = DistanceHead(args.encoder_attention_heads, args.activation_fn)
        if args.lattice_loss > 0:
            # self.lattice_head = ClassificationHead(input_dim=args.encoder_embed_dim,
            #                                     inner_dim=args.encoder_embed_dim,
            #                                     num_classes=3,
            #                                     activation_fn=self.args.pooler_activation_fn,
            #                                     pooler_dropout=self.args.pooler_dropout,
            # )
            self.lattice_head = NodeClassificationHead(input_dim=args.encoder_embed_dim,
                                                inner_dim=args.encoder_embed_dim,
                                                num_classes=6,
                                                activation_fn=self.args.pooler_activation_fn,
                                                pooler_dropout=self.args.pooler_dropout,
            )
        self.classification_heads = nn.ModuleDict()
        self.node_classification_heads = nn.ModuleDict()
        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        select_atom,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        atom_descriptor=None,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        **kwargs
    ):

        if classification_head_name is not None:
            features_only = True

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)
        def get_dist_features(dist, et):
            n_node = et.size(-1)
            gbf_feature = self.gbf(dist, et)
            # print("gbf_feature", gbf_feature.shape)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            # print("gbf_result", gbf_result.shape)
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            # print("graph_attn_bias", graph_attn_bias.shape)
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            # print("graph_attn_bias", graph_attn_bias.shape)
            return graph_attn_bias
        # print(x.shape, src_distance.shape, src_edge_type.shape)
        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        encoder_rep, encoder_pair_rep, delta_encoder_pair_rep, x_norm, delta_encoder_pair_rep_norm = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float('-inf')] = 0
        # print(encoder_rep.shape, graph_attn_bias.shape, encoder_pair_rep.shape, delta_encoder_pair_rep.shape, x_norm, delta_encoder_pair_rep_norm)
        encoder_distance = None
        encoder_coord = None
        lattice = None

        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.lm_head(encoder_rep, encoder_masked_tokens)
            if self.args.masked_coord_loss > 0:
                coords_emb = src_coord
                if padding_mask is not None:
                    atom_num = (torch.sum(1-padding_mask.type_as(x), dim=1) - 1).view(-1, 1, 1, 1)
                else:
                    atom_num = src_coord.shape[1] - 1
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                coord_update = torch.sum(coord_update, dim=2)
                encoder_coord = coords_emb + coord_update
            if self.args.masked_dist_loss > 0:
                if self.args.global_distance:
                    encoder_distance = self.glodist_head(encoder_pair_rep)
                else:
                    encoder_distance = self.dist_head(encoder_pair_rep)
            if self.args.lattice_loss > 0:
                lattice = self.lattice_head(encoder_rep[select_atom==1])
        if atom_descriptor is not None:
            model_input = torch.cat((encoder_rep, atom_descriptor.to(encoder_rep.dtype)), dim=2)
        else : 
            model_input = encoder_rep
        if classification_head_name is not None:
            if classification_head_name in self.classification_heads:
                logits = self.classification_heads[classification_head_name](model_input)
            elif classification_head_name in self.node_classification_heads:
                logits = self.node_classification_heads[classification_head_name](model_input[select_atom==1])
        elif features_only and (self.classification_heads or self.node_classification_heads):
            logits = {}
            for name, head in self.node_classification_heads.items():
                logits[name] = head(model_input[select_atom==1])
            for name, head in self.classification_heads.items():
                logits[name] = head(model_input)

        return logits, encoder_distance, encoder_coord, lattice, x_norm, delta_encoder_pair_rep_norm

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, extra_dim=0, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        input_dim=self.args.encoder_embed_dim + extra_dim
        self.classification_heads[name] = ClassificationHead(
            input_dim=input_dim,
            inner_dim=inner_dim or input_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def register_node_classification_head(
        self, name, num_classes=None, inner_dim=None, extra_dim=0, **kwargs
    ):
        """Register a classification head."""
        if name in self.node_classification_heads:
            prev_num_classes = self.node_classification_heads[name].out_proj.out_features
            prev_inner_dim = self.node_classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        input_dim=self.args.encoder_embed_dim + extra_dim
        self.node_classification_heads[name] = NodeClassificationHead(
            input_dim=input_dim,
            inner_dim=inner_dim or input_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )    

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class NodeClassificationHead(nn.Module):
    """Head for node-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x

class GlobalDistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 4)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x)
        x = (x + x.transpose(1, 2)) * 0.5
        return x

@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):  # x: (atoms, hidden),  edge_type: atoms, atoms
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class GlobalGaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):  # x: (atoms, hidden),  edge_type: atoms, atoms
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x + bias     # (bsz, n, n, 16)
        # x = x.repeat(1, 1, 1, int(self.K/x.shape[-1]))
        # replication_rules = [18, 14, 12, 12, 12, 12, 8, 8, 8, 8, 3, 3, 3, 3, 2, 2]
        # replication_rules = [66, 8, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2]
        # replication_rules = [30, 17, 17, 17, 17, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2]
        # replication_rules = [80, 32, 10, 6]
        replication_rules = [128, 0, 0, 0]
        # replication_rules = [88, 40, 0, 0]
        x = torch.repeat_interleave(x, torch.tensor(replication_rules).to(x.device), dim=-1)  # (bsz, n, n, 128)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class NumericalEmbed(nn.Module):
    def __init__(self, K=128, edge_types=1024, activation_fn='gelu'):
        super().__init__()
        self.K = K 
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        self.w_edge = nn.Embedding(edge_types, K)

        self.proj = NonLinearHead(1, K, activation_fn, hidden=2*K)
        self.ln = LayerNorm(K)

        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
        nn.init.kaiming_normal_(self.w_edge.weight)


    def forward(self, x, edge_type):    # edge_type, atoms
        x = x[:,:,:,0].squeeze(-1)
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        w_edge = self.w_edge(edge_type).type_as(x)
        edge_emb = w_edge * torch.sigmoid(mul * x.unsqueeze(-1) + bias)
        
        edge_proj = x.unsqueeze(-1).type_as(self.mul.weight)
        edge_proj = self.proj(edge_proj)
        edge_proj = self.ln(edge_proj)

        h = edge_proj + edge_emb
        h = h.type_as(self.mul.weight)
        return h
        

@register_model_architecture("unimat", "unimat")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 8)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 1024)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    args.lattice_loss = getattr(args, "lattice_loss", -1.0)


@register_model_architecture("unimat", "unimat_base")
def unimol_base_architecture(args):
    base_architecture(args)

@register_model_architecture("unimat", "unimol_large")
def unimol_base_architecture(args):
    base_architecture(args)
    args.encoder_layers = 15
