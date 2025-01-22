
import lmdb
import os
from tqdm import tqdm
import pickle
import numpy as np

def write_lmdb(outputfilename, nmr_data, nthreads=4):

    try:
        os.remove(outputfilename)
    except:
        pass
    env_new = lmdb.open(
        outputfilename,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        # max_readers=1,
        map_size=int(10e12),
    )
    txn_write = env_new.begin(write=True)
    i = 0
    for index in tqdm(range(len(nmr_data))):
        inner_output = pickle.dumps(nmr_data[index], protocol=-1)
        txn_write.put(f'{i}'.encode("ascii"), inner_output)
        i += 1
        if i % 100 == 0:
            txn_write.commit()
            txn_write = env_new.begin(write=True)
    txn_write.commit()
    env_new.close()