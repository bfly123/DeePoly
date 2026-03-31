#!/usr/bin/env python3
"""Build a DeePoly JAX feature provider from a JAX-PI checkpoint."""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_jax_runtime(
    *,
    jax_platform: str | None = None,
    cuda_visible_devices: str | None = None,
) -> None:
    if jax_platform:
        platform = str(jax_platform).strip().lower()
        if platform == "gpu":
            platform = "cuda"
        os.environ.setdefault("JAX_PLATFORMS", platform)
        os.environ.setdefault("JAX_PLATFORM_NAME", platform)
    if cuda_visible_devices is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_visible_devices)


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _pushd(path: Path):
    cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


def _resolve_runtime_path(path_value: str | None, case_dir: str | os.PathLike[str] | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if case_dir is not None:
        return (Path(case_dir).expanduser().resolve() / path).resolve()
    return path.resolve()


def _resolve_coordinate_order(coordinate_order: Iterable[int] | None, n_dim: int) -> tuple[int, ...]:
    if coordinate_order is None:
        if n_dim == 2:
            return (1, 0)
        return tuple(range(n_dim))
    order = tuple(int(index) for index in coordinate_order)
    if len(order) != n_dim:
        raise ValueError(f"coordinate_order length {len(order)} does not match n_dim={n_dim}")
    return order


def build_provider(
    config,
    model,
    case_dir,
    data_train,
    data_test,
    provider_class,
    *,
    jaxpi_root: str,
    workdir: str,
    config_path: str | None = None,
    model_file: str = "models.py",
    utils_file: str = "utils.py",
    model_class: str = "AllenCahn",
    config_function: str = "get_config",
    dataset_function: str = "get_dataset",
    coordinate_order: Sequence[int] | None = None,
    jax_platform: str | None = "cpu",
    cuda_visible_devices: str | None = "",
):
    _ = (model, case_dir, data_train, data_test)
    _configure_jax_runtime(
        jax_platform=jax_platform,
        cuda_visible_devices=cuda_visible_devices,
    )

    jaxpi_root_path = _resolve_runtime_path(jaxpi_root, case_dir)
    if jaxpi_root_path is None:
        raise ValueError("jaxpi_root must be provided for JAX-PI feature provider.")
    if str(jaxpi_root_path) not in sys.path:
        sys.path.insert(0, str(jaxpi_root_path))

    try:
        import jax.numpy as jnp
        from jaxpi.utils import restore_checkpoint
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "The active environment is missing JAX-PI runtime dependencies. "
            "Activate the JAX environment before using tools/jaxpi_feature_factory.py."
        ) from exc

    if config_path is None:
        config_file = jaxpi_root_path / "examples" / "allen_cahn" / "configs" / "pirate.py"
    else:
        config_file = _resolve_runtime_path(config_path, case_dir)
        if config_file is None:
            raise ValueError("config_path resolution failed.")

    example_dir = config_file.parent.parent
    model_module = _load_module("deepoly_jaxpi_models", example_dir / model_file)
    utils_module = _load_module("deepoly_jaxpi_utils", example_dir / utils_file)
    config_module = _load_module("deepoly_jaxpi_config", config_file)

    jaxpi_config = getattr(config_module, config_function)()
    with _pushd(example_dir):
        u_ref, t_star, x_star = getattr(utils_module, dataset_function)()
    u0 = u_ref[0, :]

    if jaxpi_config.use_pi_init:
        jaxpi_config.arch.pi_init = jnp.zeros((jaxpi_config.arch.hidden_dim, jaxpi_config.arch.out_dim))

    model_type = getattr(model_module, model_class)
    with _pushd(example_dir):
        model_instance = model_type(jaxpi_config, u0, t_star, x_star)

    workdir_path = _resolve_runtime_path(workdir, case_dir)
    if workdir_path is None:
        raise ValueError("workdir must be provided for JAX-PI feature provider.")
    ckpt_path = workdir_path / jaxpi_config.wandb.name / "ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"JAX-PI checkpoint directory not found: {ckpt_path}")

    with _pushd(example_dir):
        model_instance.state = restore_checkpoint(model_instance.state, str(ckpt_path))

    params = model_instance.state.params
    order = _resolve_coordinate_order(coordinate_order, int(getattr(config, "n_dim", 2)))

    def hidden_fn(point):
        z = jnp.asarray(point, dtype=jnp.float64)[jnp.asarray(order, dtype=jnp.int32)]
        hidden, _ = model_instance.state.apply_fn(params, z)
        return hidden

    def output_fn(point):
        z = jnp.asarray(point, dtype=jnp.float64)[jnp.asarray(order, dtype=jnp.int32)]
        _, output = model_instance.state.apply_fn(params, z)
        return output

    feature_dim = int(getattr(jaxpi_config.arch, "hidden_dim", 0) or 0)
    return provider_class(hidden_fn=hidden_fn, output_fn=output_fn, feature_dim=feature_dim or None)
