#!/usr/bin/env bash
# Sioux Falls intervention benchmark end to end; outputs go to runs/ (about 6 minutes on one CPU core).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
"$PY" download_tntp.py
"$PY" prepare.py
"$PY" calibrate.py --model cubic
"$PY" calibrate.py --model bpr4
"$PY" results.py
"$PY" certify_with_mate.py
