#!/bin/bash
set -e

[ -f lammps.done ] || {
    if [ -f md.restart.* ]; then mpirun -np 2 lmp_mpi -i input.lmp -p 1 1 -v restart 1; else mpirun -np 2 lmp_mpi -i input.lmp -p 1 1 -v restart 0; fi
    touch lammps.done
}
