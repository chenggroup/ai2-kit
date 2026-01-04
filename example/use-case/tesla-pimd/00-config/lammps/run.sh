#!/bin/bash
set -e

IPI_INPUT=input.xml
nbeads=4
ngpu=4

WKDIR="$PWD"

[ -f lammps.done ] || {
srun --environment=dpmd-lmp-plmd-ipi \
     --container-workdir=${WKDIR} \
     --cpu-bind=socket \
     --ntasks=1 \
     --cpus-per-task=1 \
     --exclusive \
     --mem-per-cpu 4000MB \
     --gres=gpu:0 \
     ${JOBREPORT} -- i-pi $IPI_INPUT &> log.ipi &

echo "Launch i-pi code ..."
sleep 20
## Launch LAMMPS for each bead and cyclically dispatch to GPU cards 
for ((ibead=0; ibead<nbeads; ibead++))
do

echo "Launch a lammps job for ibead: ${ibead} on gpu card"
srun --environment=dpmd-lmp-plmd-ipi \
     --container-workdir=${WKDIR} \
     --container-env=OMP_NUM_THREADS,TF_INTRA_OP_PARALLELISM_THREADS,TF_INTER_OP_PARALLELISM_THREADS \
     --cpu-bind=socket \
     --ntasks=1 \
     --cpus-per-task=${OMP_NUM_THREADS} \
     --exclusive \
     --mem-per-cpu 4000MB \
     --gres=gpu:1 \
     lmp -v restart 0 -var lammpsid ${ibead} -i lammps.in 1>> lammps-${ibead}.log 2>> lammps-${ibead}.err &

done
wait



    touch lammps.done
}
