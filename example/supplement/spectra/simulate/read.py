import numpy as np

box=np.zeros((500000,9),dtype=np.float32)
coord=np.zeros((500000,360,3),dtype=np.float32)

with open('trj.xyz', 'r') as f:
    i = 0
    while i < 500000:
        j=0
        lines = [f.readline() for _ in range(369)]
        if not lines[0]:
            break
        box[i,0] = 12.42
        box[i,4] = 12.42
        box[i,8] = 50
        for line in lines[9:]:
            data = line.split()
            coord[i,j,0] = float(data[1])
            coord[i,j,1] = float(data[2])
            coord[i,j,2] = float(data[3])
            j = j+1
        i = i+1
   
np.save('./data/set.000/box.npy', box)
np.save('./data/set.000/coord.npy', coord)