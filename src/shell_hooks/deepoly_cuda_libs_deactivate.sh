#!/usr/bin/env bash

if [ "${_DEEPPOLY_CUDA_LIBS_ACTIVE:-}" != "1" ]; then
  return 0 2>/dev/null || exit 0
fi

if [ -n "${_DEEPPOLY_OLD_LD_LIBRARY_PATH+x}" ]; then
  export LD_LIBRARY_PATH="${_DEEPPOLY_OLD_LD_LIBRARY_PATH}"
  unset _DEEPPOLY_OLD_LD_LIBRARY_PATH
else
  unset LD_LIBRARY_PATH
fi

unset _DEEPPOLY_CUDA_LIBS_ACTIVE
