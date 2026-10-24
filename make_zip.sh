#!/usr/bin/env bash
# Package the code ZIP for upload (no data, no virtualenv, no cache).
set -euo pipefail
cd "$(dirname "$0")"
OUT="${1:-decode-detect-code.zip}"
rm -f "$OUT"
zip -q -X -r "$OUT" README.md LICENSE requirements.txt run.sh make_zip.sh AI_USE.md THIRD_PARTY.md NOTES.md CASES.md \
    src -x '*/__pycache__/*' '*.pyc'
unzip -l "$OUT" | tail -1
ls -l "$OUT"
