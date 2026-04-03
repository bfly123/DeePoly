from __future__ import annotations

import os
from typing import Any, Dict, Optional

from src.spotter_runtime.backends.func_fitting_common import build_func_fitting_context, supports_func_fitting_builtin
from src.spotter_runtime.backends.pde_program_common import (
    build_symbolic_pde_context,
    supports_symbolic_pde_builtin,
)
from src.spotter_runtime.config_support import extract_provider_options


def _default_backend_device(config) -> str:
    raw = str(getattr(config, "device", "cpu") or "cpu").lower()
    if raw == "cuda":
        return "gpu"
    return raw


def _configure_jax_environment(provider_options: Dict[str, Any], config) -> None:
    jax_platform = provider_options.get("jax_platform")
    if not jax_platform:
        jax_platform = "cuda" if str(getattr(config, "device", "cpu") or "cpu").lower() == "cuda" else "cpu"
    platform = str(jax_platform).strip().lower()
    if platform == "gpu":
        platform = "cuda"
    os.environ.setdefault("JAX_PLATFORMS", platform)
    os.environ.setdefault("JAX_PLATFORM_NAME", platform)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    cuda_visible_devices = provider_options.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cuda_visible_devices))
    elif platform == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def create_builtin_external_provider(
    *,
    backend: str,
    provider_cfg: Optional[Dict[str, Any]],
    config,
    data_train: Optional[Dict[str, Any]] = None,
):
    backend_name = str(backend or "torch").lower()
    if backend_name not in {"jax", "paddle"}:
        return None

    provider_options = extract_provider_options(provider_cfg)

    if supports_func_fitting_builtin(config, data_train=data_train):
        program = build_func_fitting_context(config, data_train=data_train, provider_cfg=provider_cfg)
        if backend_name == "jax":
            _configure_jax_environment(provider_options, config)
            from src.spotter_runtime.backends.jax_func_fitting_provider import TrainableJaxFuncFittingProvider

            return TrainableJaxFuncFittingProvider(program=program)
        from src.spotter_runtime.backends.paddle_func_fitting_provider import TrainablePaddleFuncFittingProvider

        paddle_device = str(provider_options.get("device") or _default_backend_device(config))
        return TrainablePaddleFuncFittingProvider(program=program, device=paddle_device)

    if supports_symbolic_pde_builtin(config, data_train=data_train):
        program = build_symbolic_pde_context(config, data_train=data_train, provider_cfg=provider_cfg)
        if backend_name == "jax":
            _configure_jax_environment(provider_options, config)
            from src.spotter_runtime.backends.jax_linear_pde_provider import TrainableJaxSymbolicPDEProvider

            return TrainableJaxSymbolicPDEProvider(program=program)
        from src.spotter_runtime.backends.paddle_linear_pde_provider import TrainablePaddleSymbolicPDEProvider

        paddle_device = str(provider_options.get("device") or _default_backend_device(config))
        return TrainablePaddleSymbolicPDEProvider(program=program, device=paddle_device)

    return None
