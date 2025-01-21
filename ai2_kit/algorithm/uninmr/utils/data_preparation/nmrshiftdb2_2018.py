from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import numpy as np
import os
from tqdm import tqdm
import lmdb

from write_lmdb import write_lmdb

def get_atomic_numbers(mol):
    atomic_numbers = []  
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        atomic_numbers.append(atomic_number)
    return atomic_numbers

def get_element_symbols(mol):
    element_symbols = []  
    for atom in mol.GetAtoms():
        element_symbol = atom.GetSymbol() 
        element_symbols.append(element_symbol)
    return element_symbols


mode = 'test'
ele = 'C'
file_path = f"/vepfs/fs_users/xufanjie/data/NMR/dataset/kuhn2019/{ele}.pickle"


with open(file_path, 'rb') as file:
    data = pickle.load(file)

nmrdata = data[f'{mode}_df']
nmrdata_list = []

for mol_id in nmrdata.index:
    mol = nmrdata['rdmol'][mol_id]
    
    try:
        Chem.SanitizeMol(mol)
    except:
        continue
    
    if '.' in Chem.MolToSmiles(mol):
        continue

    atoms = get_element_symbols(mol)
    assert mol.GetNumConformers() >= 1, "The molecule must have at least one conformer."
    coordinates = np.array(mol.GetConformer().GetPositions())
    atoms_target = np.array([0.0] * len(atoms))
    atoms_target_mask = np.array([0] * len(atoms))


    for atom_index, shift in nmrdata['value'][mol_id][0].items():
        atoms_target[atom_index] = shift
        atoms_target_mask[atom_index] = 1
    
    ret = {
            'atoms': atoms,
            'coordinates': coordinates,
            'atoms_target': atoms_target,
            'atoms_target_mask': atoms_target_mask,
    }

    nmrdata_list.append(ret)

write_lmdb(f'/vepfs/fs_users/xufanjie/NMRNet/data/nmrshiftdb2_2018/{ele}/{mode}.lmdb', nmrdata_list)