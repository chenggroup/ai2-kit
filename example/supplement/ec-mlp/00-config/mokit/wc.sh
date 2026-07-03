#!/bin/bash                                                                            
set -e

[ -f mokit.done ] || {
    /public/home/jpqiu/conda/env/mokit-py311/bin/python wc.py > cp2k_mlwf.out 2>&1
    touch mokit.done
}

