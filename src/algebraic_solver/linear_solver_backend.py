import ctypes
import glob
import os
import site
import sys
import warnings


def _discover_nvidia_library_dirs() -> list[str]:
    """Collect pip-installed NVIDIA runtime library directories for the active env."""
    candidates = []
    roots = [sys.prefix]
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass

    for root in roots:
        pattern = os.path.join(root, "nvidia", "*", "lib")
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                candidates.append(os.path.abspath(path))

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def _prepend_library_dirs_to_env(library_dirs: list[str]) -> None:
    if not library_dirs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [path for path in current.split(":") if path]
    merged = list(dict.fromkeys(library_dirs + parts))
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


def _shared_object_candidates(library_dirs: list[str]) -> list[str]:
    preferred_patterns = [
        "libcudart.so*",
        "libnvrtc.so*",
        "libnvJitLink.so*",
        "libcublas.so*",
        "libcublasLt.so*",
        "libcusparse.so*",
        "libcusolver.so*",
        "libcufft.so*",
        "libcurand.so*",
        "libcudnn.so*",
        "libnccl.so*",
        "libnvToolsExt.so*",
        "libcupti.so*",
    ]
    discovered = []
    for directory in library_dirs:
        for pattern in preferred_patterns:
            discovered.extend(sorted(glob.glob(os.path.join(directory, pattern))))
    return list(dict.fromkeys(discovered))


def _preload_nvidia_cuda_libraries() -> None:
    """Preload pip-installed NVIDIA shared objects so CuPy can resolve them."""
    library_dirs = _discover_nvidia_library_dirs()
    if not library_dirs:
        return

    _prepend_library_dirs_to_env(library_dirs)

    loaded_any = False
    for shared_object in _shared_object_candidates(library_dirs):
        try:
            ctypes.CDLL(shared_object, mode=ctypes.RTLD_GLOBAL)
            loaded_any = True
        except OSError:
            continue

    if loaded_any:
        os.environ.setdefault("CUDA_PATH", sys.prefix)


try:
    _preload_nvidia_cuda_libraries()
    import cupy as cp
    from cupyx.scipy.sparse import linalg as cplinalg

    CUPY_AVAILABLE = True
    GPU_OUT_OF_MEMORY_ERROR = cp.cuda.memory.OutOfMemoryError
except ImportError:
    cp = None
    cplinalg = None
    CUPY_AVAILABLE = False
    GPU_OUT_OF_MEMORY_ERROR = RuntimeError
    warnings.warn(
        "CuPy未Installation，GPU加速Function将Unavailable。"
        "Using pip install cupy-cuda11x Installation（According toCUDAVersion选择）"
    )
