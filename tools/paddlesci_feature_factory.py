#!/usr/bin/env python3
"""Build a DeePoly PaddleScience feature provider from a pretrained Paddle model."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    pretrained_model_path: str,
    paddlesci_root: str | None = None,
    input_keys: Sequence[str] = ("t", "x"),
    output_keys: Sequence[str] = ("u",),
    output_key: str | None = None,
    num_blocks: int = 3,
    hidden_size: int = 256,
    network_type: str = "piratenet",
    bottleneck_size: int | None = None,
    bottleneck_activation: str | None = "tanh",
    activation: str = "tanh",
    periods=None,
    fourier=None,
    random_weight=None,
    coordinate_order: Sequence[int] | None = None,
    device: str | None = None,
    dtype: str = "float64",
):
    _ = (model, data_train, data_test)

    paddlesci_root_path = _resolve_runtime_path(paddlesci_root, case_dir)
    if paddlesci_root_path is not None and str(paddlesci_root_path) not in sys.path:
        sys.path.insert(0, str(paddlesci_root_path))

    try:
        import paddle
        import ppsci
        from ppsci.utils import save_load
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "The active environment is missing Paddle / PaddleScience runtime dependencies. "
            "Activate the Paddle environment before using tools/paddlesci_feature_factory.py."
        ) from exc

    if device is not None:
        paddle.set_device(str(device))
    compute_dtype = str(dtype or "float64").lower()
    if compute_dtype not in {"float32", "float64"}:
        raise ValueError(f"Unsupported Paddle feature dtype: {dtype}")

    normalized_network_type = str(network_type or "piratenet").strip().lower()
    if normalized_network_type in {"bottleneck_piratenet", "piratenet_bottleneck"} or bottleneck_size is not None:
        from tools.paddlesci_bottleneck_piratenet import build_bottleneck_piratenet

        model_instance = build_bottleneck_piratenet(
            input_keys=tuple(input_keys),
            output_keys=tuple(output_keys),
            num_blocks=int(num_blocks),
            hidden_size=int(hidden_size),
            bottleneck_size=int(bottleneck_size if bottleneck_size is not None else hidden_size),
            bottleneck_activation=bottleneck_activation,
            activation=activation,
            periods=periods,
            fourier=fourier,
            random_weight=random_weight,
        )
    else:
        model_instance = ppsci.arch.PirateNet(
            input_keys=tuple(input_keys),
            output_keys=tuple(output_keys),
            num_blocks=int(num_blocks),
            hidden_size=int(hidden_size),
            activation=activation,
            periods=periods,
            fourier=fourier,
            random_weight=random_weight,
        )

    weights_path = _resolve_runtime_path(pretrained_model_path, case_dir)
    if weights_path is None:
        raise ValueError("pretrained_model_path must be provided for Paddle feature provider.")
    save_load.load_pretrain(model_instance, str(weights_path))
    if compute_dtype == "float64":
        model_instance = model_instance.to(dtype="float64")
    model_instance.eval()

    order = _resolve_coordinate_order(coordinate_order, int(getattr(config, "n_dim", len(input_keys))))
    output_key = output_key or tuple(output_keys)[0]

    def _prepare_inputs(points):
        pts = points
        if not isinstance(pts, paddle.Tensor):
            pts = paddle.to_tensor(pts, dtype=compute_dtype)
        elif str(pts.dtype).lower() != f"paddle.{compute_dtype}":
            pts = paddle.cast(pts, compute_dtype)
        if len(pts.shape) == 1:
            pts = paddle.unsqueeze(pts, axis=0)
        ordered_points = paddle.concat([pts[:, idx : idx + 1] for idx in order], axis=1)
        raw_input_dict = {
            key: ordered_points[:, i : i + 1] for i, key in enumerate(tuple(input_keys))
        }
        transformed_input_dict = raw_input_dict
        if model_instance._input_transform is not None:
            transformed_input_dict = model_instance._input_transform(transformed_input_dict)
        if model_instance.periods:
            transformed_input_dict = model_instance.period_emb(transformed_input_dict)
        return raw_input_dict, transformed_input_dict

    def hidden_batch_fn(points):
        _, transformed_input_dict = _prepare_inputs(points)
        features = model_instance.concat_to_tensor(
            transformed_input_dict,
            model_instance.input_keys,
            axis=-1,
        )
        if model_instance.fourier:
            features = model_instance.fourier_emb(features)
        else:
            features = model_instance.linear_emb(features)

        if hasattr(model_instance, "hidden_features_tensor"):
            hidden = model_instance.hidden_features_tensor(features)
        else:
            embed_u = model_instance.embed_u(features)
            embed_v = model_instance.embed_v(features)
            hidden = features
            for block in model_instance.blocks:
                hidden = block(hidden, embed_u, embed_v)
        return hidden

    def output_batch_fn(points):
        raw_input_dict, _ = _prepare_inputs(points)
        hidden = hidden_batch_fn(points)
        output = model_instance.last_fc(hidden)
        output = model_instance.split_to_dict(output, model_instance.output_keys, axis=-1)
        if model_instance._output_transform is not None:
            output = model_instance._output_transform(raw_input_dict, output)
        return output[output_key]

    return provider_class(
        hidden_batch_fn=hidden_batch_fn,
        output_batch_fn=output_batch_fn,
        device=device,
        feature_dim=int(bottleneck_size) if bottleneck_size is not None else int(hidden_size),
        dtype=compute_dtype,
    )
