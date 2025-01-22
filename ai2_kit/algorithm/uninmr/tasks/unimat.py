# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import torch
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    TokenizeDataset,
    RightPadDataset2D,
    LMDBDataset
)
from ..data import (
    KeyDataset,
    ToTorchDataset,
    MaskPointsDataset,
    DistanceDataset,
    GlobalDistanceDataset,
    EdgeTypeDataset,
    RightPadDataset3D,
    PrependAndAppend2DDataset,
    PrependAndAppend3DDataset,
    RightPadDataset2D0,
    LatticeNormalizeDataset,
    LatticeMatrixNormalizeDataset,
    RemoveHydrogenDataset,
    CroppingDataset,
    NormalizeDataset,
    SelectTokenDataset,
    FilterDataset,
)
from unicore.tasks import UnicoreTask, register_task
from ...uninmr.utils import parse_select_atom


logger = logging.getLogger(__name__)


@register_task("unimat")
class UniMatTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1.0,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=512,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=8.0,
            help="distance threshold for distance loss",
        )
        parser.add_argument(
            "--minkowski-p",
            type=float,
            default=2.0,
            help="minkowski p for distance loss",
        )
        parser.add_argument(
            "--random-choice",
            type=float,
            default=-1.0,
            help="random choice atom to predict lattice",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--global-distance",
            action="store_true",
            help="use global distance",
        )
        parser.add_argument(
            "--selected-atom",
            default="All",
            help="select atom: All or H or H&C&F...",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.selected_token = parse_select_atom(self.dictionary, args.selected_atom)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        if self.args.remove_hydrogen:
            dataset = RemoveHydrogenDataset(dataset, 'atoms', 'coordinates')
        dataset = CroppingDataset(dataset, self.args.seed, 'atoms', 'coordinates', self.args.max_atoms)
        dataset = NormalizeDataset(dataset, 'coordinates')

        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        # atoms_target_mask_dataset = KeyDataset(dataset, "atoms_target_mask")
        select_atom_dataset = SelectTokenDataset(token_dataset=token_dataset, selected_token=self.selected_token, random_choice=self.args.random_choice)

        coord_dataset = KeyDataset(dataset, "coordinates")

        expand_dataset = MaskPointsDataset(
            token_dataset,
            coord_dataset,
            self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=self.mask_idx,
            noise_type=self.args.noise_type,
            noise=self.args.noise,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
        )

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
        encoder_target_dataset = KeyDataset(expand_dataset, "targets")
        encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")

        token_dataset = PrependAndAppend(token_dataset, self.dictionary.bos(), self.dictionary.pad())
        select_atom_dataset = PrependAndAppend(select_atom_dataset, self.dictionary.bos(), self.dictionary.pad())

        src_dataset = PrependAndAppend(encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos())
        tgt_dataset = PrependAndAppend(encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad())
        coord_dataset = ToTorchDataset(coord_dataset, 'float32')

        if self.args.global_distance:
            lattice_dataset = LatticeNormalizeDataset(dataset, 'abc', 'angles')
            lattice_dataset = ToTorchDataset(lattice_dataset, 'float32')
            lattice_matrix_dataset = LatticeMatrixNormalizeDataset(dataset, 'lattice_matrix')
            logger.info("use global distance: {}".format(self.args.global_distance))
            encoder_distance_dataset = GlobalDistanceDataset(encoder_coord_dataset, lattice_matrix_dataset, p=self.args.minkowski_p)
            encoder_distance_dataset = PrependAndAppend3DDataset(encoder_distance_dataset, 0.0)
            encoder_distance_dataset = RightPadDataset3D(encoder_distance_dataset, pad_idx=0)

            distance_dataset = GlobalDistanceDataset(coord_dataset, lattice_matrix_dataset, p=self.args.minkowski_p)
            distance_dataset = PrependAndAppend3DDataset(distance_dataset, 0.0)
            distance_dataset = RightPadDataset3D(distance_dataset, pad_idx=0)
        else:
            lattice_dataset = None
            encoder_distance_dataset = DistanceDataset(encoder_coord_dataset, p=self.args.minkowski_p)
            encoder_distance_dataset = PrependAndAppend2DDataset(encoder_distance_dataset, 0.0)
            encoder_distance_dataset = RightPadDataset2D(encoder_distance_dataset, pad_idx=0)

            distance_dataset = DistanceDataset(coord_dataset, p=self.args.minkowski_p)
            distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
            distance_dataset = RightPadDataset2D(distance_dataset, pad_idx=0)

        encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)

        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))

        net_input = {
                "select_atom": RightPadDataset(
                    select_atom_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "src_coord": RightPadDataset2D0(
                    encoder_coord_dataset,
                    pad_idx=0,
                ),
                "src_distance": encoder_distance_dataset,
                "src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
        }
        target = {
            "tokens_target": RightPadDataset(tgt_dataset, pad_idx=self.dictionary.pad()),
            "distance_target": distance_dataset,
            "coord_target": RightPadDataset2D0(coord_dataset, pad_idx=0),
            "lattice_target": lattice_dataset,
            }
        dataset = {"net_input": net_input, "target": target}
        dataset = NestedDictionaryDataset(
            dataset
        )
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        return model