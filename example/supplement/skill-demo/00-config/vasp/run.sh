#!/bin/bash
# VASP single-point labeling job runner (template)
# Variables substituted by omb combo:
#   @POSCAR_FILE   - absolute path to the frame's POSCAR file
#   @INCAR_FILE    - absolute path to 00-config/vasp/INCAR
#   @KPOINTS_FILE  - absolute path to 00-config/vasp/KPOINTS
#   @POTCAR_FILE   - absolute path to 00-config/vasp/POTCAR (user-provided)
set -e

ln -sf @POSCAR_FILE  POSCAR
ln -sf @INCAR_FILE   INCAR
ln -sf @KPOINTS_FILE KPOINTS
ln -sf @POTCAR_FILE  POTCAR

[ -f vasp.done ] || {
    mpirun vasp_std > output 2>&1 \
        && touch success.flag    \
        || touch error.flag
    touch vasp.done
}
