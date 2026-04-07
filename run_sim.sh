#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/results/sim_100kb_auto}"
MINIMAP2_EXECUTABLE="${MINIMAP2_EXECUTABLE:-/Users/mschatz/miniforge3/bin/minimap2}"

export PYTHONPATH="$ROOT_DIR/src"

python3 -m anicompare.cli_run_experiment \
  --simulate-length 100000 \
  --output-dir "$OUTPUT_DIR" \
  --replicates 3 \
  --mutation-rates 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.15 0.18 0.20 \
  --substitution-scale 1.0 \
  --insertion-scale 0.0 \
  --deletion-scale 0.0 \
  --k 21 \
  --sketch-mode exact \
  --workers 3 \
  --minimap2-threads 4 \
  --minimap2-executable "$MINIMAP2_EXECUTABLE" \
  --force \
  --seed 17

python3 -m anicompare.cli_plot_results \
  --input "$OUTPUT_DIR/master_results.tsv" \
  --output-dir "$OUTPUT_DIR/plots"
