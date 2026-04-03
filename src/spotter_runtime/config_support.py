from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from src.spotter_runtime.backend_registry import backend_supports_inprocess


_PROVIDER_META_KEYS = {
    "factory_file",
    "factory_function",
    "runtime",
    "python",
    "python_executable",
    "required_modules",
    "allow_torch_training",
    "options",
    "factory_kwargs",
    "preset",
    "profile",
}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_spotter_resource_path(path_value: str | None, case_dir: str | os.PathLike[str] | None = None) -> Path | None:
    if path_value is None:
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    if raw.startswith("deepoly://"):
        rel = raw[len("deepoly://"):].lstrip("/")
        return (get_repo_root() / rel).resolve()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    if case_dir is not None:
        return (Path(case_dir).expanduser().resolve() / path).resolve()
    return path.resolve()


def _backend_python_fallback(backend: str) -> str:
    backend = str(backend or "").lower()
    candidates = []
    if backend == "paddle":
        candidates.append(Path.home() / "anaconda3/envs/paddle_ac/bin/python")
    elif backend == "jax":
        candidates.append(Path.home() / "anaconda3/envs/ML/bin/python")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def resolve_python_executable(python_value: str | None, *, backend: str) -> str:
    if python_value is None or not str(python_value).strip():
        return _backend_python_fallback(backend)
    raw = str(python_value).strip()
    if raw.startswith("env:"):
        env_name = raw.split(":", 1)[1].strip()
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value
        return _backend_python_fallback(backend)
    return raw


def extract_provider_options(provider_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    cfg = dict(provider_cfg or {})
    merged: Dict[str, Any] = {}
    legacy_factory_kwargs = cfg.get("factory_kwargs")
    if isinstance(legacy_factory_kwargs, dict):
        merged.update(legacy_factory_kwargs)
    direct_options = cfg.get("options")
    if isinstance(direct_options, dict):
        merged.update(direct_options)
    for key, value in cfg.items():
        if key in _PROVIDER_META_KEYS:
            continue
        merged.setdefault(key, value)
    return merged


def normalize_spotter_feature_config(
    provider_cfg: Dict[str, Any] | None,
    *,
    backend: str,
) -> Dict[str, Any]:
    backend_name = str(backend or "torch").lower()
    cfg = dict(provider_cfg or {})
    normalized: Dict[str, Any] = {
        key: value
        for key, value in cfg.items()
        if key in _PROVIDER_META_KEYS - {"factory_kwargs", "preset", "profile"}
    }
    options = extract_provider_options(cfg)
    if options:
        normalized["options"] = options
    if backend_name != "torch":
        default_runtime = "subprocess" if backend_name == "jax" else "auto"
        normalized.setdefault("runtime", default_runtime)
        normalized.setdefault("allow_torch_training", False)
    if backend_name == "jax":
        normalized.setdefault("required_modules", ["jax", "optax"])
    elif backend_name == "paddle":
        normalized.setdefault("required_modules", ["paddle"])
    return normalized


def normalize_spotter_backend_config(config: Any) -> None:
    backend = str(getattr(config, "spotter_feature_backend", "torch") or "torch").lower()
    config.spotter_feature = normalize_spotter_feature_config(
        getattr(config, "spotter_feature", {}) or {},
        backend=backend,
    )


def resolve_required_modules(provider_cfg: Dict[str, Any] | None, *, backend: str) -> Tuple[str, ...]:
    cfg = dict(provider_cfg or {})
    explicit = cfg.get("required_modules")
    if explicit is None:
        return tuple()
    if isinstance(explicit, str):
        return (explicit,)
    return tuple(str(item) for item in explicit)


def resolve_spotter_runtime(runtime_value: str | None, *, backend: str, provider_cfg: Dict[str, Any] | None = None) -> str:
    runtime = str(runtime_value or "auto").lower()
    if runtime != "auto":
        return runtime
    required_modules = resolve_required_modules(provider_cfg, backend=backend)
    if backend_supports_inprocess(backend, required_modules=required_modules or None):
        return "inprocess"
    return "subprocess"
