from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


def _build_hidden_features(x: float, y: float, hidden_dim: int) -> jnp.ndarray:
    theta = jnp.pi * (x + 1.0)
    features = [
        1.0 - y,
        y * (1.0 - y),
        jnp.sin(theta),
        jnp.cos(theta),
        y * jnp.sin(2.0 * theta),
        y * jnp.cos(2.0 * theta),
        jnp.tanh(2.0 * x),
        (1.0 - y) * jnp.tanh(2.0 * x),
    ]
    if hidden_dim <= len(features):
        return jnp.asarray(features[:hidden_dim], dtype=jnp.float64)
    padded = features + [jnp.zeros((), dtype=jnp.float64)] * (hidden_dim - len(features))
    return jnp.asarray(padded, dtype=jnp.float64)


def _build_output(x: float, y: float) -> jnp.ndarray:
    theta = jnp.pi * (x + 1.0)
    u_bottom = x * x * jnp.cos(jnp.pi * x)
    value = (
        -1.0
        + (1.0 - y) * (u_bottom + 1.0)
        + 0.08 * y * (1.0 - y) * jnp.sin(theta)
        + 0.04 * y * (1.0 - y) * jnp.cos(2.0 * theta)
        + 0.05 * y * jnp.tanh(2.0 * x)
    )
    return jnp.asarray([value], dtype=jnp.float64)


def build_provider(
    config,
    model,
    case_dir,
    data_train,
    data_test,
    provider_class,
    *,
    hidden_dim: int = 8,
):
    _ = (config, model, case_dir, data_train, data_test)

    def hidden_fn(point):
        x = point[0]
        y = point[1]
        return _build_hidden_features(x, y, hidden_dim)

    def output_fn(point):
        x = point[0]
        y = point[1]
        return _build_output(x, y)

    return provider_class(hidden_fn=hidden_fn, output_fn=output_fn, feature_dim=hidden_dim)
