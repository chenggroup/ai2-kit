#!/bin/bash
set -e

[ -f train.done ] || {
    dp train input.json
    touch train.done
}


dp freeze -o frozen_model.pb
dp compress -i frozen_model.pb -o compress.pb