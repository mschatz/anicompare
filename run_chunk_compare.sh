#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <reference_fasta> <query_fasta> <chunk_length> [output_dir]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFERENCE_FASTA="$1"
QUERY_FASTA="$2"
CHUNK_LENGTH="$3"
REFERENCE_STEM="$(basename "$REFERENCE_FASTA")"
REFERENCE_STEM="${REFERENCE_STEM%%.*}"
QUERY_STEM="$(basename "$QUERY_FASTA")"
QUERY_STEM="${QUERY_STEM%%.*}"
OUTPUT_DIR="${4:-$ROOT_DIR/results/${REFERENCE_STEM}_vs_${QUERY_STEM}_${CHUNK_LENGTH}bp}"
MINIMAP2_EXECUTABLE="${MINIMAP2_EXECUTABLE:-/Users/mschatz/miniforge3/bin/minimap2}"

export PYTHONPATH="$ROOT_DIR/src"

python3 -m anicompare.cli_run_experiment \
  --reference-fasta "$REFERENCE_FASTA" \
  --query-fasta "$QUERY_FASTA" \
  --output-dir "$OUTPUT_DIR" \
  --analysis-mode reference_chunks \
  --chunk-length "$CHUNK_LENGTH" \
  --replicates 1 \
  --k 21 \
  --sketch-mode exact \
  --workers 4 \
  --minimap2-threads 2 \
  --minimap2-executable "$MINIMAP2_EXECUTABLE" \
  --minimap2-preset asm20 \
  --force

python3 -m anicompare.cli_plot_results \
  --input "$OUTPUT_DIR/master_results.tsv" \
  --output-dir "$OUTPUT_DIR/plots" \
  --report-mode observed
