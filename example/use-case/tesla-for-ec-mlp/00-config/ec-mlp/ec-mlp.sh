#!/bin/bash
set -e

[ -f ec-mlp.done ] || {
    dp train ec-mlp.json
    touch ec-mlp.done
}


dp freeze -o ec-mlp.pb
dp compress -i ec-mlp.pb -o ec-mlp-compress.pb
