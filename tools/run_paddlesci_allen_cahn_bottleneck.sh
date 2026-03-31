#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/tmp/paddle_bottleneck_h256_b20_adam}"
DEVICE="${DEVICE:-gpu}"
OPTIM="${OPTIM:-adam}"

if [[ $# -ge 1 ]]; then
  OUTPUT_DIR="$1"
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("paddle", "ppsci", "scipy") if importlib.util.find_spec(name) is None]
if missing:
    print("Current Python environment is missing required packages: " + ", ".join(missing), file=sys.stderr)
    print("Activate the Paddle environment first, for example: conda activate paddle_ac", file=sys.stderr)
    sys.exit(1)
PY

echo "Running PaddleScience Allen-Cahn bottleneck training"
echo "  python: ${PYTHON_BIN}"
echo "  output: ${OUTPUT_DIR}"
echo "  device: ${DEVICE}"
echo "  optim : ${OPTIM}"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/train_paddlesci_allen_cahn_bottleneck.py" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --optim "${OPTIM}" \
  --hidden-size 256 \
  --bottleneck-size 20 \
  --activation tanh \
  --bottleneck-activation tanh \
  --fourier-dim 256 \
  --fourier-scale 2.0 \
  --period-x 2.0 \
  --rwf-mean 1.0 \
  --rwf-std 0.1 \
  --lr 1e-3 \
  --gamma 0.9 \
  --decay-steps 5000 \
  --epochs 300 \
  --iters-per-epoch 1000 \
  --batch-size 8192 \
  --eval-batch-size 4096 \
  --save-freq 10 \
  --eval-freq 10 \
  --log-freq 100 \
  --causal-chunks 32 \
  --causal-tol 1.0 \
  --grad-norm-update-freq 1000 \
  --grad-norm-momentum 0.9 \
  --official-pretrained-model "${ROOT_DIR}/tmp/PaddleScience_20260329/examples/allen_cahn/allen_cahn_piratenet_soap_pretrained.pdparams"
