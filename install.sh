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

SCRIPT_PATH="$("$PYTHON_BIN" -m src.install_env script-path)"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: installation completed but CLI entry script was not created: $SCRIPT_PATH" >&2
  exit 1
fi

echo "Installed DeePoly CLI."
echo "CLI entry: $SCRIPT_PATH"
echo "Usage:"
echo "  deepoly ."
