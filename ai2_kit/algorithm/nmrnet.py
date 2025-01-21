from argparse import Namespace
from itertools import product
from scipy.spatial import distance_matrix

import numpy as np
import argparse

import ase.io
from ase import Atoms

import torch
from torch.utils.data import Dataset, DataLoader

from unicore import checkpoint_utils

from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    TokenizeDataset,
    RightPadDataset2D,
)

from .uninmr.utils import parse_select_atom, TargetScaler
from .uninmr.models import UniMatModel
from .uninmr.data import (
    KeyDataset,
    IndexDataset,
    ToTorchDataset,
    DistanceDataset,
    GlobalDistanceDataset,
    EdgeTypeDataset,
    RightPadDataset3D,
    PrependAndAppend2DDataset,
    PrependAndAppend3DDataset,
    RightPadDataset2D0,
    LatticeMatrixNormalizeDataset,
    CroppingDataset,
    NormalizeDataset,
    TargetScalerDataset,
    SelectTokenDataset,
    FilterDataset,
)


from ai2_kit.core.log import get_logger
logger = get_logger(__name__)


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def single_rcut(atoms: Atoms, rcut=6):
    rcut_atoms = []
    rcut_coords = []
    rcut_targets = []
    rcut_masks = []

    ks = [0, 1, -1, 2, -2]
    pbc_repeat = 5 * 5 * 5

    atom = atoms.get_chemical_symbols()
    pbc_atoms = atom * pbc_repeat

    pos = atoms.get_positions()
    pbc_pos = np.tile(pos, (pbc_repeat, 1, 1))
    lattice_matrix = (atoms.cell)
    pbc_matrix = np.array(list(product(ks, repeat=3)))
    pbc_pos += np.dot(pbc_matrix, lattice_matrix).reshape(pbc_repeat, -1, 3)
    pbc_pos = pbc_pos.reshape(-1, 3)
    dist_pbc = distance_matrix(pbc_pos, pos, 2).astype(np.float32)

    for i, element in enumerate(atom):
        dist_mask = (dist_pbc.reshape(-1, pos.shape[0])[:, i] < rcut)
        rcut_atoms.append(np.array(pbc_atoms)[dist_mask].tolist())
        rcut_coords.append(pbc_pos[dist_mask])
        rcut_target = [0] * (len(atoms) * pbc_repeat)
        rcut_targets.append(np.array(rcut_target)[dist_mask])
        rcut_mask = [0] * (len(atoms) * pbc_repeat)
        rcut_mask[i] = 1
        rcut_masks.append(np.array(rcut_mask)[dist_mask])
    return rcut_atoms, rcut_coords, rcut_targets, rcut_masks


def get_args(model_path, dict_path, saved_dir, selected_atom='H', nmr_type='solid'):
    args = Namespace()

    args.model_path=model_path
    args.dict_path=dict_path
    args.selected_atom = selected_atom
    args.nmr_type=nmr_type
    args.saved_dir=saved_dir

    args.encoder_layers = 8
    args.encoder_embed_dim = 512
    args.encoder_ffn_embed_dim = 2048
    args.encoder_attention_heads = 64
    args.dropout = 0.1
    args.emb_dropout = 0.1
    args.attention_dropout = 0.1
    args.activation_dropout = 0.0
    args.pooler_dropout = 0.0
    args.max_seq_len = 1024
    args.activation_fn = "gelu"
    args.pooler_activation_fn = "tanh"
    args.post_ln = False
    args.masked_token_loss = -1.0
    args.masked_coord_loss = -1.0
    args.masked_dist_loss = -1.0
    args.x_norm_loss = -1.0
    args.delta_pair_repr_norm_loss = -1.0
    args.lattice_loss = -1.0
    args.encoder_layers = 15
    args.num_classes=1
    args.atom_descriptor=0
    args.classification_head_name='nmr_head'
    args.global_distance=0
    args.gaussian_kernel = True
    args.max_atoms=512
    args.max_seq_len=1024
    args.seed=1
    args.batch_size=16
    args.required_batch_size_multiple=1
    args.num_workers=8
    args.data_buffer_size=10
    args.log_format='simple'
    args.log_interval=50
    return args


def load_dataset(atoms: Atoms, args: Namespace, dictionary:Dictionary, target_scaler:TargetScaler):
    selected_token = parse_select_atom(dictionary, args.selected_atom)
    nmr_type = args.nmr_type

    if nmr_type == 'solid':
        rcut_atoms, rcut_coords, rcut_targets, rcut_masks = single_rcut(atoms, rcut=6)
    elif nmr_type == 'liquid':
        raise NotImplementedError("Liquid NMR prediction is not supported yet.")
    else:
        raise ValueError(f"Invalid nmr_type: {nmr_type}")

    rcut_list = []
    for i in range(len(rcut_atoms)):
        ret = {}
        ret['atoms'] = rcut_atoms[i]
        ret['coordinates'] = rcut_coords[i]
        ret['atoms_target'] = rcut_targets[i]
        ret['atoms_target_mask'] = rcut_masks[i]
        rcut_list.append(ret)

    dataset = ListDataset(rcut_list)
    matid_dataset = IndexDataset(dataset)
    dataset = CroppingDataset(dataset, args.seed, "atoms", "coordinates", args.max_atoms)
    dataset = NormalizeDataset(dataset, "coordinates")

    token_dataset = KeyDataset(dataset, "atoms")
    token_dataset = TokenizeDataset(token_dataset, dictionary, max_seq_len=args.max_seq_len)
    atoms_target_mask_dataset = KeyDataset(dataset, "atoms_target_mask")
    select_atom_dataset = SelectTokenDataset(token_dataset=token_dataset, token_mask_dataset=atoms_target_mask_dataset, selected_token=selected_token)
    filter_list = [0 if torch.all(select_atom_dataset[i]==0) else 1 for i in range(len(select_atom_dataset))]

    dataset = FilterDataset(dataset, filter_list)
    matid_dataset = FilterDataset(matid_dataset, filter_list)
    token_dataset = FilterDataset(token_dataset, filter_list)
    select_atom_dataset = FilterDataset(select_atom_dataset, filter_list)

    coord_dataset = KeyDataset(dataset, "coordinates")

    def PrependAndAppend(dataset, pre_token, app_token):
        dataset = PrependTokenDataset(dataset, pre_token)
        return AppendTokenDataset(dataset, app_token)

    token_dataset = PrependAndAppend(token_dataset, dictionary.bos(), dictionary.eos())
    select_atom_dataset = PrependAndAppend(select_atom_dataset, dictionary.pad(), dictionary.pad())

    coord_dataset = ToTorchDataset(coord_dataset, 'float32')

    if args.global_distance:
        lattice_matrix_dataset = LatticeMatrixNormalizeDataset(dataset, 'lattice_matrix')
        logger.info("use global distance: {}".format(args.global_distance))
        distance_dataset = GlobalDistanceDataset(coord_dataset, lattice_matrix_dataset)
        distance_dataset = PrependAndAppend3DDataset(distance_dataset, 0.0)
        distance_dataset = RightPadDataset3D(distance_dataset, pad_idx=0)
    else:
        distance_dataset = DistanceDataset(coord_dataset)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)
        distance_dataset = RightPadDataset2D(distance_dataset, pad_idx=0)

    coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
    edge_type = EdgeTypeDataset(token_dataset, len(dictionary))
    tgt_dataset = KeyDataset(dataset, "atoms_target")
    tgt_dataset = TargetScalerDataset(tgt_dataset, target_scaler, args.num_classes)
    tgt_dataset = ToTorchDataset(tgt_dataset, dtype='float32')
    tgt_dataset = PrependAndAppend(tgt_dataset, dictionary.pad(), dictionary.pad())

    return NestedDictionaryDataset(
            {
                "net_input": {
                    "select_atom": RightPadDataset(
                        select_atom_dataset,
                        pad_idx=dictionary.pad(),
                    ),
                    "src_tokens": RightPadDataset(
                        token_dataset,
                        pad_idx=dictionary.pad(),
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


def load_model(args, dictionary):
    state = checkpoint_utils.load_checkpoint_to_cpu(args.model_path)
    state['model'] = {
        (key.replace('classification_heads', 'node_classification_heads')
         if key.startswith('classification_heads') else key): value
        for key, value in state['model'].items()
    }
    model = UniMatModel(args, dictionary)  # type: ignore
    model.register_node_classification_head(
        args.classification_head_name,
        num_classes=args.num_classes,
        extra_dim=args.atom_descriptor,
    )
    model.load_state_dict(state["model"], strict=False)
    return model


def predict(model: UniMatModel, dataloader: DataLoader,
            classification_head_name, num_classes, target_scaler: TargetScaler):
    model.eval()
    with torch.no_grad():
        all_predicts = []
        for batch in dataloader:
            net_output = model(**{k.replace('net_input.', ''): v for k, v in batch.items() if k.startswith('net_input.')},
                            features_only=True,
                            classification_head_name=classification_head_name)
            predict = target_scaler.inverse_transform(
                net_output[0].view(-1, num_classes).data.cpu()
            ).astype('float32')
            all_predicts.append(predict)
        final_predicts = np.concatenate(all_predicts)
    return final_predicts.reshape(-1).reshape(-1,4).mean(axis=1)


def predict_cli(data_file: str, model_path: str, dict_path: str, saved_dir: str,
                selected_atom, nmr_type, use_cuda=False, cuda_device_id=None):

    args = get_args(model_path, dict_path, saved_dir,
                    selected_atom=selected_atom, nmr_type=nmr_type)
    if use_cuda:
        torch.cuda.set_device(cuda_device_id)

    dictionary = Dictionary.load(args.dict_path)
    target_scaler = TargetScaler(args.saved_dir)

    atoms = ase.io.read(data_file, index=0)
    assert isinstance(atoms, Atoms), "data_file must be a single ASE Atoms object"

    dataset = load_dataset(atoms, args, dictionary, target_scaler)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = load_model(args, dictionary)
    model.half()
    if use_cuda:
        model.cuda()
    else:
        model.float()
    result = predict(model, dataloader,
                     classification_head_name=args.classification_head_name,
                     num_classes=args.num_classes,
                     target_scaler=target_scaler)
    print(result)

