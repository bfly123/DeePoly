#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  echo "ERROR: pip is not available in the current Python environment: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"

echo "Validating current environment with: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
"$PYTHON_BIN" -m src.install_env check

echo "Installing DeePoly CLI into the current environment"
"$PYTHON_BIN" -m pip install -e "$ROOT_DIR"

ENV_PREFIX="$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')"
if [[ -d "$ENV_PREFIX/conda-meta" || -n "${CONDA_PREFIX:-}" ]]; then
  ACTIVATE_DIR="$ENV_PREFIX/etc/conda/activate.d"
  DEACTIVATE_DIR="$ENV_PREFIX/etc/conda/deactivate.d"
  mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"
  install -m 0644 "$ROOT_DIR/src/shell_hooks/deepoly_cuda_libs_activate.sh" \
    "$ACTIVATE_DIR/deepoly_cuda_libs.sh"
  install -m 0644 "$ROOT_DIR/src/shell_hooks/deepoly_cuda_libs_deactivate.sh" \
    "$DEACTIVATE_DIR/deepoly_cuda_libs.sh"
  echo "Installed CUDA runtime hook for conda environment: $ENV_PREFIX"
fi

SCRIPT_PATH="$("$PYTHON_BIN" -m src.install_env script-path)"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: installation completed but CLI entry script was not created: $SCRIPT_PATH" >&2
  exit 1
fi

echo "Installed DeePoly CLI."
echo "CLI entry: $SCRIPT_PATH"
echo "Usage:"
echo "  deepoly ."
