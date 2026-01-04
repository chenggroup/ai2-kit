#!/bin/bash
set -e

[ -f lammps.done ] || {
    # check if restart files are created
    ls md.restart.* &>/dev/null && RESTART=1 || RESTART=0
    lmp -i lammps.in -v restart $RESTART

    touch lammps.done
}
