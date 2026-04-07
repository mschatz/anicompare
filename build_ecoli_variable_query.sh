#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/results/ecoli_variable_query}"
REFERENCE_FASTA="${REFERENCE_FASTA:-$ROOT_DIR/data/ecoli.fa.gz}"
CHUNK_LENGTH="${CHUNK_LENGTH:-10000}"
MIN_RATE="${MIN_RATE:-0.0}"
MAX_RATE="${MAX_RATE:-0.2}"
SEED="${SEED:-17}"

export PYTHONPATH="$ROOT_DIR/src"

python3 -m anicompare.cli_build_variable_query \
  --reference-fasta "$REFERENCE_FASTA" \
  --output-fasta "$OUTPUT_DIR/query.fa" \
  --metadata-json "$OUTPUT_DIR/query_metadata.json" \
  --chunk-length "$CHUNK_LENGTH" \
  --min-rate "$MIN_RATE" \
  --max-rate "$MAX_RATE" \
  --seed "$SEED"
