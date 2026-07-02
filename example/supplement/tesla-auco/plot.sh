#!/bin/bash
set -e

python dp-test.py ./30-dp-test/test
python model-devi-plot.py "./20-workdir/iter-*/screening/stats.tsv" --out-file model-devi.png
