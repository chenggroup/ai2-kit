from mokit.lib.gaussian import pbc_loc
import numpy as np
from ase import Atoms


def read_bulk_indices(filename, element='Cu'):
    positions = []
    symbols = []
    with open(filename) as f:
        lines = f.readlines()
        coord_started = False
        for line in lines:
            line = line.strip()
            if '&COORD' in line:
                coord_started = True
                continue
            if coord_started:
                if line.startswith('&') or line == '':
                    break
                parts = line.split()
                if len(parts) == 4:
                    symbol = parts[0]
                    x, y, z = map(float, parts[1:])
                    symbols.append(symbol)
                    positions.append([x, y, z])
    
    atoms = Atoms(symbols=symbols, positions=positions)
    
    # 获取所有不等于指定元素的原子index
    non_element_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s != element]
    return np.array(non_element_indices, dtype=np.int32)

# 示例：读取文件并排除Cu元素
data_file_path= "@DATA_FILE"
proj_list = read_bulk_indices(data_file_path, element='Cu')
print(proj_list)

pbc_loc(molden='./cp2k-MOS-1_0.molden',box=data_file_path,wannier_xyz="wannier.xyz",proj_list = proj_list, conv_tol=1e-6)

