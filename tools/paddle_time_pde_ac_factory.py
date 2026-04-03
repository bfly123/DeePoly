#!/usr/bin/env python3
"""Build a trainable Paddle Allen-Cahn time-PDE spotter provider for DeePoly."""

from __future__ import annotations


def build_provider(config, model, case_dir, data_train, data_test, provider_class, *, device: str | None = None, learning_rate: float | None = None, epochs_adam: int | None = None):
    _ = (model, case_dir, data_test, provider_class)

    from src.spotter_runtime.backends.paddle_ac_time_pde_provider import TrainablePaddleACTimePDEProvider

    return TrainablePaddleACTimePDEProvider(config=config, data_train=data_train, device=device, learning_rate=learning_rate, epochs_adam=epochs_adam)
