#!/bin/bash
set -e

if [ ! -f cp2k.done ]; then
    if [ ! -f cp2k_diag.done ]; then
        echo "Running cp2k_diag.inp ..."
        mpirun cp2k.psmp -i cp2k_diag.inp &> output_diag
        touch cp2k_diag.done
    fi

    echo "Running cp2k.inp ..."
    mpirun cp2k.psmp -i cp2k.inp &> output
    touch cp2k.done
fi
