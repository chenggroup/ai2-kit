#!/usr/bin/env bash
[ -f lammps.done ] || {
    bash engine.sh
    touch lammps.done
}
