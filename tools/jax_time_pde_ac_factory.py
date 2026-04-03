#!/usr/bin/env python3
"""Build a trainable JAX Allen-Cahn time-PDE spotter provider for DeePoly."""

from __future__ import annotations

import os


def build_provider(config, model, case_dir, data_train, data_test, provider_class, *, jax_platform: str | None = None, cuda_visible_devices: str | None = None, learning_rate: float | None = None, epochs_adam: int | None = None):
    _ = (model, case_dir, data_test, provider_class)
    if jax_platform:
        platform = str(jax_platform).strip().lower()
        if platform == "gpu":
            platform = "cuda"
        os.environ.setdefault("JAX_PLATFORMS", platform)
        os.environ.setdefault("JAX_PLATFORM_NAME", platform)
    if cuda_visible_devices is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cuda_visible_devices))

    from src.spotter_runtime.backends.jax_ac_time_pde_provider import TrainableJaxACTimePDEProvider

    return TrainableJaxACTimePDEProvider(config=config, data_train=data_train, learning_rate=learning_rate, epochs_adam=epochs_adam)
