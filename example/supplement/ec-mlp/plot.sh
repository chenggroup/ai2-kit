#!/bin/bash
set -e

#python dp-test.py ./30-dp-test/test
python model-devi-plot.py "./02-workdir/iter-00*/screening/water/stats.tsv" --out-file model-devi.png

