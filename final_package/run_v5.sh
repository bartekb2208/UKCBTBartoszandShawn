#!/usr/bin/env bash
# Reproduce predictions.csv for the Model Phi entry: a 50/50 rank blend of two models.
#   ours/   model A  (LightGBM + ExtraTrees on behavioural features)
#   shawn/  model B  (scenario-reconstruction model, two-stage LightGBM)
# Needs the organiser data next to this script (override with TRAIN=... TEST=...):
#   decode-and-detect-training-data/train/   decode-and-detect-test-data/
# Usage:  PY=python3 ./run_v5.sh      (PY = the Python 3.14.6 interpreter with requirements.txt installed)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
TRAIN="$(cd "${TRAIN:-decode-and-detect-training-data/train}" && pwd)"
TEST="$(cd "${TEST:-decode-and-detect-test-data}" && pwd)"
PY="${PY:-python3}"
WORK="$ROOT/work"
mkdir -p "$WORK"

echo "[1/3] model A (ours/)"
(cd ours && "$PY" src/predict.py --train "$TRAIN" --test "$TEST" --out "$WORK/ours.csv")
echo "[2/3] model B (shawn/)"
(cd shawn && "$PY" run.py --train "$TRAIN" --test "$TEST" --out "$WORK/shawn_out")
echo "[3/3] blend 50/50 and validate"
"$PY" blend_v5.py "$WORK/ours.csv" "$WORK/shawn_out/submission.csv" 0.5 predictions.csv "$TEST"
echo "done: $ROOT/predictions.csv"
