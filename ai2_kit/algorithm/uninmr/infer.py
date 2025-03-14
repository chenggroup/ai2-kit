from argparse import Namespace
from io import StringIO
from itertools import product
from scipy.spatial import distance_matrix
import os

import numpy as np

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

from .utils import parse_select_atom, TargetScaler
from .models import UniMatModel
from .data import (
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
from ai2_kit.core.util import resolve_path

logger = get_logger(__name__)


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def extend_cells(atoms: Atoms, rcut=6):
    """
    Extend cells for periodic boundary conditions
    """

    ks = [0, 1, -1, 2, -2]
    pbc_repeat = len(ks) ** 3

    atom = atoms.get_chemical_symbols()
    pbc_atoms = atom * pbc_repeat

    pos = atoms.get_positions()
    pbc_pos = np.tile(pos, (pbc_repeat, 1, 1))
    lattice_matrix = (atoms.cell)
    pbc_matrix = np.array(list(product(ks, repeat=3)))
    pbc_pos += np.dot(pbc_matrix, lattice_matrix).reshape(pbc_repeat, -1, 3)
    pbc_pos = pbc_pos.reshape(-1, 3)
    dist_pbc = distance_matrix(pbc_pos, pos, 2).astype(np.float32)

    cells = []
    for i, e in enumerate(atom):
        dist_mask = (dist_pbc.reshape(-1, pos.shape[0])[:, i] < rcut)
        rcut_target = [0] * (len(atoms) * pbc_repeat)
        rcut_mask = [0] * (len(atoms) * pbc_repeat)
        rcut_mask[i] = 1
        cell = {
            'atoms': np.array(pbc_atoms)[dist_mask].tolist(),
            'coordinates': pbc_pos[dist_mask],
            'atoms_target': np.array(rcut_target)[dist_mask],
            'atoms_target_mask': np.array(rcut_mask)[dist_mask],
        }
        cells.append(cell)
    return cells


def get_args(model_path, dict_path, saved_dir, selected_atom='H', nmr_type='solid'):
    args = Namespace()

    args.model_path=model_path
    args.dict_path=dict_path
    args.selected_atom = selected_atom
    args.nmr_type=nmr_type
    args.saved_dir=saved_dir  # this turn out to be unused, just keep it for compatibility

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
    """
    Load dataset for NMRNet prediction
    """
    selected_token = parse_select_atom(dictionary, args.selected_atom)
    nmr_type = args.nmr_type

    if nmr_type == 'solid':
        atoms_info = extend_cells(atoms, rcut=6)
    elif nmr_type == 'liquid':
        ret = {
            'atoms': atoms.get_chemical_symbols(),
            'coordinates': atoms.get_positions(),
            'atoms_target': np.array([0] * len(atoms)),
            'atoms_target_mask': np.array([1] * len(atoms)),
        }
        atoms_info = [ret]
    else:
        raise ValueError(f"Invalid nmr_type: {nmr_type}")

    dataset = ListDataset(atoms_info)
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
    """
    Load model from checkpoint
    """
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


def smiles_to_atoms(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)  # type: ignore
    AllChem.UFFOptimizeMolecule(mol) # type: ignore
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = mol.GetConformer().GetPositions()
    atoms = Atoms(symbols=symbols, positions=positions)
    return atoms


def predict(model: UniMatModel, dataloader: DataLoader,
            classification_head_name, num_classes, target_scaler: TargetScaler):
    """
    Predict NMR Spectrum
    """
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
    return final_predicts


def predict_cli(model_path: str, dict_path: str, scaler_path: str,
                selected_atom: str, nmr_type: str, use_cuda=False, cuda_device_id=None,
                smiles: str = '', data_file: str = '', data: str = '', format=None,
                return_xyz=False):
    """
    Command line interface for NMRNet prediction.

    You can provide input data with one of `data_file`, `data` or `smiles`.

    :param model_path: path to the model checkpoint, e.g 'model.pt'
    :param dict_path: path to the dictionary file, e.g 'dict.txt'
    :param scaler_path: path to the scaler file, e.g 'target_scaler.ss'
    :param selected_atom: selected atom for prediction, e.g 'H'
    :param nmr_type: type of NMR prediction, should be 'solid' or 'liquid'
    :param use_cuda: whether to use GPU for prediction, default is False
    :param cuda_device_id: GPU device id, default is None, required when use_cuda is True
    :param data_file: path to the input data file, which should be able to parse by ASE
    :param data: input data string, default is '', you can provide data directly
    :param smiles: SMILES string for prediction, default is ''
    :param format: format of the input data file, default is None, you can find the supported format in ASE: https://wiki.fysik.dtu.dk/ase/ase/io/io.html
    """
    model_path = resolve_path(model_path)
    dict_path = resolve_path(dict_path)
    scaler_path = resolve_path(scaler_path)

    if data_file:
        data_file = resolve_path(data_file)
        atoms = ase.io.read(data_file, index=0, format=format)  # type: ignore
    elif data:
        atoms = ase.io.read(StringIO(data), index=0, format=format)  # type: ignore
    elif smiles:
        atoms = smiles_to_atoms(smiles)
    else:
        raise ValueError("data_file or smiles must be provided")

    scaler_path = os.path.abspath(scaler_path)

    scaler_dir = os.path.dirname(scaler_path)
    scaler_file = os.path.basename(scaler_path)

    args = get_args(model_path, dict_path, scaler_dir,
                    selected_atom=selected_atom, nmr_type=nmr_type)
    if use_cuda:
        torch.cuda.set_device(cuda_device_id)

    dictionary = Dictionary.load(args.dict_path)
    dictionary.add_symbol("[MASK]", is_special=True)
    target_scaler = TargetScaler(scaler_dir, scaler_file)

    assert isinstance(atoms, Atoms), "data_file must be a single ASE Atoms object"
    dataset = load_dataset(atoms, args, dictionary, target_scaler)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = load_model(args, dictionary)
    model.half()
    if use_cuda:
        model.cuda()
    else:
        model.float()
    d = predict(model, dataloader,
                classification_head_name=args.classification_head_name,
                num_classes=args.num_classes,
                target_scaler=target_scaler)
    if return_xyz:
        f = StringIO()
        ase.io.write(f, atoms, format='extxyz')
        xyz = f.getvalue()
        return d, xyz
    else:
        return d
