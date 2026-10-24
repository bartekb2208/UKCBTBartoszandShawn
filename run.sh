#!/usr/bin/env bash
# Reproduce predictions.csv from the organiser data.
# Expects decode-and-detect-training-data/train/ and test-data/ next to this script.
set -euo pipefail
cd "$(dirname "$0")"
TRAIN="${TRAIN:-decode-and-detect-training-data/train}"
TEST="${TEST:-decode-and-detect-test-data}"
OUT="${OUT:-predictions.csv}"
PY="${PY:-python3}"
"$PY" src/predict.py --train "$TRAIN" --test "$TEST" --out "$OUT"
