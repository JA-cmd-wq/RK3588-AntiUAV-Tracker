#!/usr/bin/env bash
# Run the tracker pipeline on the RK3588 board.
#   PYTHON            python interpreter with numpy, opencv-python and rknn-toolkit-lite2 (default: python3)
#   EXTRA_PYTHONPATH  optional extra site-packages directory prepended to PYTHONPATH
# Model paths default to ../models/{fear,yolo}; override with --yolo-model/--template-model/--search-model
# or the ANTI_UAV_MODELS environment variable.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
if [[ -n "${EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${EXTRA_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$PYTHON" pipeline.py "$@"
