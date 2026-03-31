from pathlib import Path

from src.algebraic_solver import linear_solver_backend as backend


def test_discover_nvidia_library_dirs_finds_active_env_runtime_paths(tmp_path, monkeypatch):
    env_root = tmp_path / "env"
    site_root = env_root / "lib" / "python3.11" / "site-packages"
    cublas_dir = site_root / "nvidia" / "cublas" / "lib"
    cusolver_dir = site_root / "nvidia" / "cusolver" / "lib"
    cublas_dir.mkdir(parents=True)
    cusolver_dir.mkdir(parents=True)

    monkeypatch.setattr(backend.sys, "prefix", str(env_root))
    monkeypatch.setattr(backend.site, "getsitepackages", lambda: [str(site_root)])

    discovered = backend._discover_nvidia_library_dirs()

    assert discovered == [
        str(cublas_dir.resolve()),
        str(cusolver_dir.resolve()),
    ]


def test_shared_object_candidates_preserve_preload_order(tmp_path):
    runtime_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
    nvrtc_dir = tmp_path / "nvidia" / "cuda_nvrtc" / "lib"
    solver_dir = tmp_path / "nvidia" / "cusolver" / "lib"
    for directory in [runtime_dir, nvrtc_dir, solver_dir]:
        directory.mkdir(parents=True)

    (runtime_dir / "libcudart.so.12").write_text("", encoding="utf-8")
    (nvrtc_dir / "libnvrtc.so.12").write_text("", encoding="utf-8")
    (solver_dir / "libcusolver.so.11").write_text("", encoding="utf-8")

    candidates = backend._shared_object_candidates(
        [str(runtime_dir), str(nvrtc_dir), str(solver_dir)]
    )

    assert candidates == [
        str(runtime_dir / "libcudart.so.12"),
        str(nvrtc_dir / "libnvrtc.so.12"),
        str(solver_dir / "libcusolver.so.11"),
    ]


def test_prepend_library_dirs_to_env_keeps_new_paths_first(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/one:/existing/two")

    backend._prepend_library_dirs_to_env(["/new/one", "/existing/one", "/new/two"])

    assert backend.os.environ["LD_LIBRARY_PATH"] == (
        "/new/one:/existing/one:/new/two:/existing/two"
    )
