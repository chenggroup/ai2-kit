import numpy as np
import json
from write_lmdb import write_lmdb

def parse_geo_file(file_path):
    molecule_ids_list = []
    atoms_list = []
    coordinates_list = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0

        while i < len(lines):
            num_atoms = int(lines[i].strip())
            i += 1
            molecule_id = int(lines[i].split('_')[1])
            i += 1  

            atoms = []
            coordinates = []

            for _ in range(num_atoms):
                atom_data = lines[i].split()
                atom_type = atom_data[0]
                atom_coords = np.array([float(coord) for coord in atom_data[1:]])
                
                atoms.append(atom_type)
                coordinates.append(atom_coords)

                i += 1

            molecule_ids_list.append(molecule_id)
            atoms_list.append(atoms)
            coordinates_list.append(np.array(coordinates))

    return molecule_ids_list, atoms_list, coordinates_list


def parse_nmr_file(file_path):
    molecule_ids_list = []
    atoms_list = []
    nmrdata_list = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0

        while i < len(lines):
            num_atoms = int(lines[i].strip())
            i += 1
            molecule_id = int(lines[i].split('_')[1])
            i += 1  

            atoms = []
            nmrdata = []

            for _ in range(num_atoms):
                atom_data = lines[i].split()
                atom_type = atom_data[0]
                atom_shift = np.array([float(shift) for shift in atom_data[1:]])
                
                atoms.append(atom_type)
                nmrdata.append(atom_shift)

                i += 1

            molecule_ids_list.append(molecule_id)
            atoms_list.append(atoms)
            nmrdata_list.append(np.array(nmrdata))

    return molecule_ids_list, atoms_list, nmrdata_list


file_path = '/vepfs/fs_users/xufanjie/data/NMR/rawdata/dataset_qm9/SI_DFT_geo.xyz'
molecule_ids_geo, atoms_geo, coordinates = parse_geo_file(file_path)

file_path = '/vepfs/fs_users/xufanjie/data/NMR/rawdata/dataset_qm9/SI_DFT_NMR.txt'
molecule_ids_nmr, atoms_nmr, nmrdata = parse_nmr_file(file_path)

testid_path = "/vepfs/fs_users/xufanjie/NMRNet/data/QM9-NMR/testid.json"

with open(testid_path, 'r') as file:
    testid = json.load(file)

train_data = []
test_data = []
pahse = 0
pahsename = ['gas', 'ccl4', 'thf', 'acetone', 'methanol', 'dmso']


for i in range(len(molecule_ids_nmr)):

    ret = {
    'atoms': atoms_geo[i],
    'coordinates': np.array(coordinates[i]),
    'atoms_target': np.array(nmrdata[i][:,pahse]),
    'atoms_target_mask': np.array([1]*len(atoms_geo[i])),
    }
    if i in testid:
        test_data.append(ret)
    else:
        train_data.append(ret)

ele = 'All'
mode = 'train'
write_lmdb(f'/vepfs/fs_users/xufanjie/NMRNet/data/QM9-NMR/{pahsename[pahse]}/{ele}/{mode}.lmdb', train_data)
mode = 'valid'
write_lmdb(f'/vepfs/fs_users/xufanjie/NMRNet/data/QM9-NMR/{pahsename[pahse]}/{ele}/{mode}.lmdb', test_data)