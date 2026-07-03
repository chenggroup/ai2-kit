#!/bin/bash
set -e

ln -sf @DATA_FILE coord_n_cell.inc

[ -f cp2k.done ] || {
    mpirun cp2k.psmp -i cp2k.inp &> output && touch success.flag || touch error.flag
    rm -f *.wfn || true
    touch cp2k.done
}
