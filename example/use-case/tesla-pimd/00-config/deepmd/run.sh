#!/bin/bash
set -e

[ -f train.done ] || {
srun --environment=dpmd-lmp-plmd \
     --container-workdir=$PWD \
     --container-env=OMP_NUM_THREADS,TF_INTRA_OP_PARALLELISM_THREADS,TF_INTER_OP_PARALLELISM_THREADS \
     --cpu-bind=socket \
     --ntasks=1 \
     --gres=gpu:1 \
     dp train input.json -v DEBUG -l dptrain.log
    touch train.done
}


srun --environment=dpmd-lmp-plmd \
     --container-workdir=$PWD \
     --container-env=OMP_NUM_THREADS,TF_INTRA_OP_PARALLELISM_THREADS,TF_INTER_OP_PARALLELISM_THREADS \
     --cpu-bind=socket \
     --ntasks=1 \
     --gres=gpu:1 \
     dp freeze -o frozen_model.pb
srun --environment=dpmd-lmp-plmd \
     --container-workdir=$PWD \
     --container-env=OMP_NUM_THREADS,TF_INTRA_OP_PARALLELISM_THREADS,TF_INTER_OP_PARALLELISM_THREADS \
     --cpu-bind=socket \
     --ntasks=1 \
     --gres=gpu:1 \
     dp compress -i frozen_model.pb -o compress.pb
