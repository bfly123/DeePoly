#!/usr/bin/env bash

if [ -n "${_DEEPPOLY_CUDA_LIBS_ACTIVE:-}" ]; then
  return 0 2>/dev/null || exit 0
fi

export _DEEPPOLY_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  _deepoly_nvidia_lib_dirs="$("${CONDA_PREFIX}/bin/python" - <<'PY'
import glob
import os
import site
import sys

paths = []
roots = [sys.prefix]
try:
    roots.extend(site.getsitepackages())
except Exception:
    pass

for root in roots:
    for path in glob.glob(os.path.join(root, "nvidia", "*", "lib")):
        if os.path.isdir(path):
            paths.append(os.path.abspath(path))

seen = []
ordered = []
for path in paths:
    if path not in seen:
        seen.append(path)
        ordered.append(path)

print(":".join(ordered), end="")
PY
)"

  if [ -n "$_deepoly_nvidia_lib_dirs" ]; then
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then
      export LD_LIBRARY_PATH="${_deepoly_nvidia_lib_dirs}:${LD_LIBRARY_PATH}"
    else
      export LD_LIBRARY_PATH="${_deepoly_nvidia_lib_dirs}"
    fi
  fi
fi

export _DEEPPOLY_CUDA_LIBS_ACTIVE=1
