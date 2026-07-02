#!/bin/bash
set -e

ln -sf @DATA_FILE coord_n_cell.inc

[ -f warmup.done ] || {
    mpirun cp2k.psmp -i cp2k_ot_low.inp &> warmup-output || touch warmup-error.flag
    touch warmup.done
}

[ -f final.done ] || {
    mpirun cp2k.psmp -i cp2k_ot_high.inp &> output && touch success.flag || touch error.flag
    rm -f AI2KIT*
    touch final.done
}
