#!/bin/bash
set -e

[ -f cp2k.done ] || {
    mpirun cp2k.psmp -i cp2k.inp &> output
    touch cp2k.done
}
