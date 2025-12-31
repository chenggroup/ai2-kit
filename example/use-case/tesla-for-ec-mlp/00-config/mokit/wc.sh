#!/bin/bash                                                                            
set -e

[ -f mokit.done ] || {
    python wc.py > cp2k_mlwf.out 2>&1
    touch mokit.done
}

