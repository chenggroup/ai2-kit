from ase import Atoms
from itertools import product
from collections import namedtuple
from scipy.spatial import distance_matrix

import numpy as np
import argparse
import torch

from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks

from unicore.data import Dictionary

from .uninmr.utils import parse_select_atom, TargetScaler
from .uninmr.models import UniMatModel

from ai2_kit.core.log import get_logger


logger = get_logger(__name__)


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

    for index, element in enumerate(atom):
        dist_mask = (dist_pbc.reshape(-1, pos.shape[0])[:, index] < rcut)
        rcut_atoms.append(np.array(pbc_atoms)[dist_mask].tolist())
        rcut_coords.append(pbc_pos[dist_mask])
        rcut_target = [0] * (len(atoms) * pbc_repeat)
        rcut_targets.append(np.array(rcut_target)[dist_mask])
        rcut_mask = [0] * (len(atoms) * pbc_repeat)
        rcut_mask[index] = 1
        rcut_masks.append(np.array(rcut_mask)[dist_mask])
    return rcut_atoms, rcut_coords, rcut_targets, rcut_masks

def get_args():
    args = argparse.Namespace()

    args.selected_atom = 'H'
    args.model_path='./weight/cv_seed_42_fold_0/checkpoint_best.pt'
    args.dict_path='./weight/oc_limit_dict.txt'
    args.saved_dir='./weight'

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


def predict(atoms, nmr_mode, use_cuda, cuda_device_id):
    args = get_args()
    dictionary = Dictionary.load(args.dict_path)
    mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
    selected_token = parse_select_atom(dictionary, args.selected_atom)
    target_scaler=TargetScaler(args.saved_dir)

    if nmr_mode == 'solid':
        rcut_atoms, rcut_coords, rcut_targets, rcut_masks = single_rcut(atoms, rcut=6)
    elif nmr_mode == 'liquid':
        raise NotImplementedError("Liquid NMR prediction is not supported yet.")
    else:
        raise ValueError(f"Invalid nmr_mode: {nmr_mode}")

    rcut_list = []
    for i in range(len(rcut_atoms)):
        ret = {}
        ret['atoms'] = rcut_atoms[i]
        ret['coordinates'] = rcut_coords[i]
        ret['atoms_target'] = rcut_targets[i]
        ret['atoms_target_mask'] = rcut_masks[i]
        rcut_list.append(ret)

    if use_cuda:
        torch.cuda.set_device(cuda_device_id)

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

    # Move models to GPU
    model.half()
    if use_cuda:
        model.cuda()
    else:
        model.float()


