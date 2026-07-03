#!/bin/bash
set -e

[ -f dw.done ] || {
    dp train dw.json
    touch dw.done
}


dp freeze -o dw.pb
