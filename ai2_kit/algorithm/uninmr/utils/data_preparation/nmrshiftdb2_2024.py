import re
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
from rdkit import RDLogger

import statistics
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool
from write_lmdb import write_lmdb

import random
from collections import defaultdict
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action="ignore")



def get_element_symbols(mol):
    element_symbols = []  
    for atom in mol.GetAtoms():
        element_symbol = atom.GetSymbol() 
        element_symbols.append(element_symbol)
    return element_symbols

def process_single_molecule(args):
    db_id, value = args
    
    key = db_id + '_0'
    mol = nmr_data[key]['mol']
    smiles = nmr_data[key]['smiles']
    
    res = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if res != 0:
        return None  

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        return None   

    if mol.GetNumConformers() != 1:
        return None   

    atoms = get_element_symbols(mol)
    coordinates = np.array(mol.GetConformer().GetPositions())
    atoms_target = np.array([0.0] * len(atoms))
    atoms_target_mask = np.array([0] * len(atoms))

    index_shift_nmrdata = {}
    for dupmol_index in range(value+1):
        key = db_id + f'_{dupmol_index}'
        for nmrindex in range(len(nmr_data[key]['indexes_list'])):
            for indexes, shift in zip(nmr_data[key]['indexes_list'][nmrindex],nmr_data[key]['shifts_list'][nmrindex]):
                for index in indexes:
                    if index in index_shift_nmrdata:
                        index_shift_nmrdata[index].append(shift)
                    else:
                        index_shift_nmrdata[index] = [shift]

    for single_index, shifts in index_shift_nmrdata.items():
        atoms_target_mask[single_index-1] = 1
        if merge_method == "median":
            atoms_target[single_index-1] = statistics.median(shifts)
        elif merge_method == "mean":
            atoms_target[single_index-1] = statistics.mean(shifts)


    try:
        inchikey = Chem.MolToInchiKey(mol)
    except:
        inchikey = None

    return {
        'atoms': atoms,
        'coordinates': coordinates,
        'atoms_target': atoms_target,
        'atoms_target_mask': atoms_target_mask,
        'smiles': smiles,
        'db_id': db_id,
        'mol': mol,
        'inchikey': inchikey,
    }

file_path = '/vepfs/fs_users/xufanjie/data/NMR/dataset_nmrshiftdb2_2024/corrected_nmrshiftdb2.nmredata.sd'

supplier = Chem.ForwardSDMolSupplier(file_path, removeHs=False)    #

failed_mol_by_RDKit_count = 0
db_id_counter = {}
nmr_data = {}
for i, mol in enumerate(supplier):
    if mol is not None:    
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        if '.' in Chem.MolToSmiles(mol):
            continue

        smiles = mol.GetProp('NMREDATA_SMILES')

        db_id = mol.GetProp('NMREDATA_ID').split('=')[-1].replace('\\', '')
        if db_id not in db_id_counter:
            db_id_counter[db_id] = 0
        else:
            db_id_counter[db_id] += 1

        assignment_data = mol.GetPropsAsDict()['NMREDATA_ASSIGNMENT']
        assignment_pattern = r"s(\d+), ([-\d.]+), ((?:\d+, )*\d+)"
        assignment_data = re.findall(assignment_pattern, assignment_data)

        shifts_list = []
        indexes_list = []
        for prop_name in mol.GetPropNames():
            if prop_name.startswith('NMREDATA_1D_'):

                nmr_1d_data_raw = mol.GetPropsAsDict()[prop_name]
                
                nmr_1d_spectrum_pattern = r"([-\d.]+), L=s(\d+)"
                nmr_1d_data = re.findall(nmr_1d_spectrum_pattern, nmr_1d_data_raw)
                

                single_shifts_list = []
                single_indexes_list = []
                isDirty = False
                for shift, peak_index in nmr_1d_data:
                    isDirty = False
                    peak_index = int(peak_index)
                    
                    assignment_index = next((j for j, k in enumerate(assignment_data) if int(k[0]) == peak_index), None)
                    if assignment_index is None:
                        continue
                    atom_numbers = [int(x) for x in assignment_data[assignment_index][2].split(', ')]

                    if float(shift) != float(assignment_data[assignment_index][1]):
                        print(f"{db_id}_{db_id_counter[db_id]}: In molecule {prop_name}, the chemical shifts differ when the assignment index matches the peak number in 1D NMR: 1D NMR shift: {float(shift)}, assignment shift: {float(assignment_data[assignment_index][1])}")
                        continue
                    
                    nmr_1d_type_pattern = re.compile(r'NMREDATA_1D_\d+([A-Z]+)')
                    nmr_1d_type = re.findall(nmr_1d_type_pattern, prop_name)
                    for atom_index in atom_numbers:
                        atom = mol.GetAtomWithIdx(atom_index-1)
                        element_type = atom.GetSymbol()
                        if element_type != nmr_1d_type[0]:
                            isDirty = True
                            break
                    
                    if isDirty:
                        break

                    if nmr_1d_type[0] == 'H':
                        if float(shift) > 15 or  float(shift) < 0:
                            isDirty = True
                            break
                    elif nmr_1d_type[0] == 'C':
                        if float(shift) > 250 or  float(shift) < -50:
                            isDirty = True
                            break
                    elif nmr_1d_type[0] == 'F':
                        if float(shift) > 100 or  float(shift) < -250:
                            isDirty = True
                            break

                    if len(atom_numbers) != len(set(atom_numbers)):
                        print(f"{db_id}_{db_id_counter[db_id]}: In molecule {prop_name}, the same chemical shift corresponds to multiple identical indices, possibly due to incorrect numbering of equivalent atoms.")

                    single_shifts_list.append((float(shift)))
                    single_indexes_list.append(atom_numbers)


                if isDirty:
                    break
                else:
                    if ((len(single_shifts_list) == 0) or (len(single_indexes_list) == 0)):
                        continue
                    else:
                        shifts_list.append(single_shifts_list)
                        indexes_list.append(single_indexes_list)

        if ((len(shifts_list) == 0) or (len(indexes_list) == 0)):
            db_id_counter[db_id] -= 1
            if (db_id_counter[db_id] == -1):
                del db_id_counter[db_id]
            continue

        key = f'{db_id}_{db_id_counter[db_id]}' 
        nmr_data[key] = {
            'smiles': smiles,
            'mol': mol,
            'db_id': db_id,
            'shifts_list': shifts_list,
            'indexes_list': indexes_list
        }
    else:
        failed_mol_by_RDKit_count += 1



merge_method = "median"  # "median" or "mean"

non_merged_duplicate_index = -1

seed = 42

tasks = [(db_id, value) for db_id, value in db_id_counter.items()]

with Pool() as pool:
    results = pool.map(process_single_molecule, tasks)

nmrdata_list = [result for result in results if result is not None]


random.seed(42)

inchikey_groups = defaultdict(list)
for idx, item in enumerate(nmrdata_list):
    inchikey = item['inchikey']
    inchikey_groups[inchikey].append(idx)

duplicate_indices_list = [idx for indices in inchikey_groups.values() if len(indices) > 1 for idx in indices]

non_duplicate_indices = [idx for indices in inchikey_groups.values() if len(indices) == 1 for idx in indices]

random.shuffle(non_duplicate_indices)  
split_index = int(len(nmrdata_list) * 0.8) - len(duplicate_indices_list)   

train_index = non_duplicate_indices[:split_index] + duplicate_indices_list
test_index = non_duplicate_indices[split_index:]

train_data = [nmrdata_list[i] for i in train_index]
test_data = [nmrdata_list[i] for i in test_index]

valid_elements = {'C', 'H', 'O', 'N', 'F', 'B'}

train_data = []
for index in train_index:
    skip_molecule = False
    for atom_index, atom in enumerate(nmrdata_list[index]['atoms']):
        if atom not in valid_elements:
            nmrdata_list[index]['atoms_target_mask'][atom_index] = 0
        
        # Additional conditions for N and O
        if atom == 'N' and nmrdata_list[index]['atoms_target'][atom_index] < -100:
            skip_molecule = True
            break
        if atom == 'O' and nmrdata_list[index]['atoms_target'][atom_index] > 490:
            skip_molecule = True
            break

    if all(element == 0 for element in nmrdata_list[index]['atoms_target_mask']):
        print(f"{index} molecule has uncommon active elements")
    elif skip_molecule:
        print(f"{index} molecule has uncommon NMR")
    else:
        train_data.append(nmrdata_list[index])

test_data = []
for index in test_index:
    skip_molecule = False
    for atom_index, atom in enumerate(nmrdata_list[index]['atoms']):
        if atom not in valid_elements:
            nmrdata_list[index]['atoms_target_mask'][atom_index] = 0
        
        # Additional conditions for N and O
        if atom == 'N' and nmrdata_list[index]['atoms_target'][atom_index] < -100:
            skip_molecule = True
            break
        if atom == 'O' and nmrdata_list[index]['atoms_target'][atom_index] > 490:
            skip_molecule = True
            break

    if all(element == 0 for element in nmrdata_list[index]['atoms_target_mask']):
        print(f"{index} molecule has uncommon active elements")
    elif skip_molecule:
        print(f"{index} molecule has uncommon NMR")
    else:
        test_data.append(nmrdata_list[index])

ele = 'All'
mode = 'train'
write_lmdb(f'/vepfs/fs_users/xufanjie/NMRNet/data/nmrshiftdb2_2024/{ele}/{mode}.lmdb', train_data)
mode = 'valid'
write_lmdb(f'/vepfs/fs_users/xufanjie/NMRNet/data/nmrshiftdb2_2024/{ele}/{mode}.lmdb', test_data)