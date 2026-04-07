#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/results/ecoli_variable_10kb}"
REFERENCE_FASTA="${REFERENCE_FASTA:-$ROOT_DIR/data/ecoli.fa.gz}"
MINIMAP2_EXECUTABLE="${MINIMAP2_EXECUTABLE:-/Users/mschatz/miniforge3/bin/minimap2}"
CHUNK_LENGTH="${CHUNK_LENGTH:-10000}"
MIN_RATE="${MIN_RATE:-0.0}"
MAX_RATE="${MAX_RATE:-0.2}"

export PYTHONPATH="$ROOT_DIR/src"

python3 -m anicompare.cli_run_experiment \
  --reference-fasta "$REFERENCE_FASTA" \
  --output-dir "$OUTPUT_DIR" \
  --analysis-mode reference_chunks \
  --chunk-length "$CHUNK_LENGTH" \
  --variable-chunk-mutation \
  --variable-chunk-min-rate "$MIN_RATE" \
  --variable-chunk-max-rate "$MAX_RATE" \
  --replicates 1 \
  --mutation-rates 0.0 \
  --substitution-scale 1.0 \
  --insertion-scale 0.0 \
  --deletion-scale 0.0 \
  --k 21 \
  --sketch-mode exact \
  --workers 4 \
  --minimap2-threads 2 \
  --minimap2-executable "$MINIMAP2_EXECUTABLE" \
  --minimap2-preset asm20 \
  --force \
  --seed 17

python3 -m anicompare.cli_plot_results \
  --input "$OUTPUT_DIR/master_results.tsv" \
  --output-dir "$OUTPUT_DIR/plots"
