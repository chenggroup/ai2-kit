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
    LMDBDataset,
    RawLabelDataset,
)
from ..data import (
    KeyDataset,
    ConformerSampleDataset,
    TTADataset,
    IndexDataset,
    TTAIndexDataset,
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
    TargetScalerDataset,
    FoldLMDBDataset,
    StackedLMDBDataset,
    SplitLMDBDataset,
    SelectTokenDataset,
    FilterDataset,
)
from unicore.tasks import UnicoreTask, register_task
from uninmr.utils import parse_select_atom, TargetScaler
from ase import Atoms
logger = logging.getLogger(__name__)

@register_task("uninmr")
class UniNMRTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path"
        )
        parser.add_argument(
            "--saved-dir",
            help="saved dir"
        )
        parser.add_argument(
            "--classification-head-name",
            default="nmr_head",
            help="finetune downstream task name"
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers"
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
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--has-matid",
            action="store_true",
            help="whether already has matid",
        )
        parser.add_argument(
            "--conformer-augmentation",
            action="store_true",
            help="using conformer augmentation",
        )
        parser.add_argument(
            "--conf-size",
            type=int,
            default=10,
            help="conformer nums per structure",
        )
        parser.add_argument(
            "--global-distance",
            action="store_true",
            help="use global distance",
        )
        parser.add_argument(
            "--atom-descriptor",
            type=int,
            default=0,
            help="use extra atom descriptor",
        )
        parser.add_argument(
            "--selected-atom",
            default="All",
            help="select atom: All or H or H&C&F...",
        )
        parser.add_argument(
            '--split-mode',
            type=str,
            default='predefine',
            choices=['predefine', 'cross_valid', 'random', 'infer'],
        )
        parser.add_argument(
            "--nfolds",
            default=5,
            type=int,
            help="cross validation split folds"
        )
        parser.add_argument(
            "--fold",
            default=0,
            type=int,
            help='local fold used as validation set, and other folds will be used as train set'
        )
        parser.add_argument(
            "--cv-seed",
            default=42,
            type=int,
            help="random seed used to do cross validation splits"
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.selected_token = parse_select_atom(self.dictionary, args.selected_atom)
        if args.saved_dir == None:
            self.args.saved_dir = args.save_dir
        self.target_scaler = TargetScaler(args.saved_dir)

        if self.args.split_mode =='predefine':
            train_path = os.path.join(self.args.data, "train" + ".lmdb")
            self.train_dataset = LMDBDataset(train_path)
            valid_path = os.path.join(self.args.data, "valid" + ".lmdb")
            self.valid_dataset = LMDBDataset(valid_path)
            atoms_target = np.concatenate([np.array(self.train_dataset[i]['atoms_target']) for i in range(len(self.train_dataset))], axis=0)
            atoms_target_mask = np.concatenate([np.array(self.train_dataset[i]['atoms_target_mask']) for i in range(len(self.train_dataset))], axis=0)
            self.target_scaler.fit(target=atoms_target[atoms_target_mask==1].reshape(-1, self.args.num_classes), num_classes=self.args.num_classes, dump_dir=self.args.save_dir)
        elif self.args.split_mode == 'infer':
            valid_path = os.path.join(self.args.data, "valid" + ".lmdb")
            self.valid_dataset = LMDBDataset(valid_path)
        else:
            self.__init_data()

    def __init_data(self):
        data_path = os.path.join(self.args.data, 'train.lmdb')
        raw_dataset = LMDBDataset(data_path)
        atoms_target = np.concatenate([np.array(raw_dataset[i]['atoms_target']) for i in range(len(raw_dataset))], axis=0)
        atoms_target_mask = np.concatenate([np.array(raw_dataset[i]['atoms_target_mask']) for i in range(len(raw_dataset))], axis=0)

        if self.args.split_mode == 'cross_valid':
            train_folds = []
            for _fold in range(self.args.nfolds):
                if _fold == 0:
                    parent_dir = os.path.dirname(self.args.saved_dir)
                    self.target_scaler.fit(target=atoms_target[atoms_target_mask==1].reshape(-1, self.args.num_classes), num_classes=self.args.num_classes, dump_dir=parent_dir)
                    cache_fold_info = FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds).get_fold_info()
                if _fold == self.args.fold:
                    self.valid_dataset = FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds, cache_fold_info=cache_fold_info)
                if _fold != self.args.fold:
                    train_folds.append(FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds, cache_fold_info=cache_fold_info))
            self.train_dataset = StackedLMDBDataset(train_folds)
        elif self.args.split_mode == 'random':
            self.target_scaler.fit(target=atoms_target[atoms_target_mask==1].reshape(-1, self.args.num_classes), num_classes=self.args.num_classes, dump_dir=self.args.saved_dir)
            cache_fold_info = SplitLMDBDataset(raw_dataset, self.args.seed, 0).get_fold_info()
            self.train_dataset = SplitLMDBDataset(raw_dataset, self.args.seed, 0, cache_fold_info=cache_fold_info)
            self.valid_dataset = SplitLMDBDataset(raw_dataset, self.args.seed, 1, cache_fold_info=cache_fold_info)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        # self.split = split
        # if self.args.split_mode != 'predefine':
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'valid':
            dataset =self.valid_dataset
        # else:
        #     split_path = os.path.join(self.args.data, split + ".lmdb")
        #     dataset = LMDBDataset(split_path)
        #     if split == 'train':
        #         atoms_target = np.concatenate([np.array(dataset[i]['atoms_target']) for i in range(len(dataset))], axis=0)
        #         atoms_target_mask = np.concatenate([np.array(dataset[i]['atoms_target_mask']) for i in range(len(dataset))], axis=0)
        #         self.target_scaler.fit(target=atoms_target[atoms_target_mask==1].reshape(-1, self.args.num_classes), num_classes=self.args.num_classes, dump_dir=self.args.save_dir)

        if self.args.has_matid:
            matid_dataset = KeyDataset(dataset, "matid")
        else:
            matid_dataset = IndexDataset(dataset)

        if self.args.conformer_augmentation:
            if split == 'train':
                dataset = ConformerSampleDataset(dataset, self.seed, "atoms", "coordinates_list")
            else:
                dataset = TTADataset(dataset, self.seed, "atoms", "coordinates_list", self.args.conf_size)
                matid_dataset = TTAIndexDataset(matid_dataset, self.args.conf_size)
        if self.args.remove_hydrogen:
            dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates")
        dataset = CroppingDataset(dataset, self.seed, "atoms", "coordinates", self.args.max_atoms)
        dataset = NormalizeDataset(dataset, "coordinates")

        # lattice_dataset = LatticeNormalizeDataset(dataset, 'abc', 'angles')
        # lattice_dataset = ToTorchDataset(lattice_dataset, 'float32')

        token_dataset = KeyDataset(dataset, "atoms")
        token_dataset = TokenizeDataset(token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        atoms_target_mask_dataset = KeyDataset(dataset, "atoms_target_mask")
        select_atom_dataset = SelectTokenDataset(token_dataset=token_dataset, token_mask_dataset=atoms_target_mask_dataset, selected_token=self.selected_token)
        filter_list = [0 if torch.all(select_atom_dataset[i]==0) else 1 for i in range(len(select_atom_dataset))]

        dataset = FilterDataset(dataset, filter_list)
        matid_dataset = FilterDataset(matid_dataset, filter_list)
        token_dataset = FilterDataset(token_dataset, filter_list)
        select_atom_dataset = FilterDataset(select_atom_dataset, filter_list)

        coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        token_dataset = PrependAndAppend(token_dataset, self.dictionary.bos(), self.dictionary.eos())
        select_atom_dataset = PrependAndAppend(select_atom_dataset, self.dictionary.pad(), self.dictionary.pad())

        coord_dataset = ToTorchDataset(coord_dataset, 'float32')

        if self.args.global_distance:
            lattice_matrix_dataset = LatticeMatrixNormalizeDataset(dataset, 'lattice_matrix')
            logger.info("use global distance: {}".format(self.args.global_distance))
            distance_dataset = GlobalDistanceDataset(coord_dataset, lattice_matrix_dataset)
            distance_dataset = PrependAndAppend3DDataset(distance_dataset, 0.0)
            distance_dataset = RightPadDataset3D(distance_dataset, pad_idx=0)
        else:
            distance_dataset = DistanceDataset(coord_dataset)
            distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
            distance_dataset = RightPadDataset2D(distance_dataset, pad_idx=0)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        edge_type = EdgeTypeDataset(token_dataset, len(self.dictionary))

        tgt_dataset = KeyDataset(dataset, "atoms_target")
        tgt_dataset = TargetScalerDataset(tgt_dataset, self.target_scaler, self.args.num_classes)
        tgt_dataset = ToTorchDataset(tgt_dataset, dtype='float32')

        tgt_dataset = PrependAndAppend(tgt_dataset, self.dictionary.pad(), self.dictionary.pad())

        if self.args.atom_descriptor != 0:
            atomdes_dataset = KeyDataset(dataset, "atoms_descriptor")
            atomdes_dataset = ToTorchDataset(atomdes_dataset, dtype='float32')
            atomdes_dataset = PrependAndAppend(atomdes_dataset, self.dictionary.pad(), self.dictionary.pad())
            nest_dataset = NestedDictionaryDataset(
                    {
                        "net_input": {
                            "select_atom": RightPadDataset(
                                select_atom_dataset,
                                pad_idx=self.dictionary.pad(),
                            ),
                            "src_tokens": RightPadDataset(
                                token_dataset,
                                pad_idx=self.dictionary.pad(),
                            ),
                            "src_coord": RightPadDataset2D0(
                                coord_dataset,
                                pad_idx=0,
                            ),
                            "src_distance": distance_dataset,
                            "src_edge_type": RightPadDataset2D(
                                edge_type,
                                pad_idx=0,
                            ),
                            "atom_descriptor": RightPadDataset2D0(
                                atomdes_dataset,
                                pad_idx=0,
                            ),
                        },
                        "target": {
                            "finetune_target": RightPadDataset(
                                tgt_dataset,
                                pad_idx=0,
                            ),
                        },
                        "matid": matid_dataset,
                    },
                )
        else:
            nest_dataset = NestedDictionaryDataset(
                    {
                        "net_input": {
                            "select_atom": RightPadDataset(
                                select_atom_dataset,
                                pad_idx=self.dictionary.pad(),
                            ),
                            "src_tokens": RightPadDataset(
                                token_dataset,
                                pad_idx=self.dictionary.pad(),
                            ),
                            "src_coord": RightPadDataset2D0(
                                coord_dataset,
                                pad_idx=0,
                            ),
                            "src_distance": distance_dataset,
                            "src_edge_type": RightPadDataset2D(
                                edge_type,
                                pad_idx=0,
                            ),
                        },
                        "target": {
                            "finetune_target": RightPadDataset(
                                tgt_dataset,
                                pad_idx=0,
                            ),
                        },
                        "matid": matid_dataset,
                    },
                )
        if split in ["train", "train.small"]:
            nest_dataset = EpochShuffleDataset(nest_dataset, len(nest_dataset), self.args.seed)
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        model.register_node_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
            extra_dim=self.args.atom_descriptor,
        )
        return model