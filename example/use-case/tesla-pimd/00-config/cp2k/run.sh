#!/bin/bash
set -e

[ -f cpk2.done ] || {
JOBREPORT=/capstor/store/cscs/userlab/lp07/zyongbin/bin/jobreport
MPS_WRAPPER=/capstor/store/cscs/userlab/lp07/zyongbin/sbatch_scripts/mps-wrapper.sh

srun --cpu-bind=socket ${MPS_WRAPPER} ${JOBREPORT}  -- cp2k.psmp -i cp2k.inp -o output

${JOBREPORT} print jobreport_${SLURM_JOB_ID}
    touch cpk2.done
}
