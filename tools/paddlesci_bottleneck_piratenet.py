#!/usr/bin/env python3
"""Custom PaddleScience PirateNet with a narrow DeePoly feature bottleneck."""

from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union


def _ensure_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    import sys

    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root()


def _build_activation(activation_name: Optional[str], feature_dim: int):
    import paddle.nn as nn
    from ppsci.arch import activation as act_mod

    if activation_name is None:
        return None
    name = str(activation_name).strip().lower()
    if name in {"", "identity", "linear", "none"}:
        return nn.Identity()
    if name == "stan":
        return act_mod.get_activation("stan")(feature_dim)
    return act_mod.get_activation(name)


class BottleneckPirateNet:
    """Factory namespace for the actual Paddle model class."""

    @staticmethod
    def build_class():
        import paddle.nn as nn
        from ppsci.arch.mlp import PirateNet
        from ppsci.arch.mlp import RandomWeightFactorization

        class _BottleneckPirateNet(PirateNet):
            def __init__(
                self,
                input_keys: Tuple[str, ...],
                output_keys: Tuple[str, ...],
                num_blocks: int,
                hidden_size: int,
                bottleneck_size: int,
                activation: str = "tanh",
                bottleneck_activation: Optional[str] = "tanh",
                weight_norm: bool = False,
                input_dim: Optional[int] = None,
                output_dim: Optional[int] = None,
                periods: Optional[Dict[int, Tuple[float, bool]]] = None,
                fourier: Optional[Dict[str, Union[float, int]]] = None,
                random_weight: Optional[Dict[str, float]] = None,
            ):
                if int(bottleneck_size) <= 0:
                    raise ValueError(f"bottleneck_size must be positive, got {bottleneck_size}")

                super().__init__(
                    input_keys=input_keys,
                    output_keys=output_keys,
                    num_blocks=num_blocks,
                    hidden_size=hidden_size,
                    activation=activation,
                    weight_norm=weight_norm,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    periods=periods,
                    fourier=fourier,
                    random_weight=random_weight,
                )

                if hasattr(self.last_fc, "in_features"):
                    body_feature_dim = int(self.last_fc.in_features)
                else:
                    body_feature_dim = int(self.last_fc.weight.shape[0])
                final_output_dim = len(self.output_keys) if output_dim is None else int(output_dim)
                self.body_feature_dim = body_feature_dim
                self.feature_dim = int(bottleneck_size)
                self.bottleneck_activation_name = bottleneck_activation

                if random_weight:
                    self.feature_proj = RandomWeightFactorization(
                        body_feature_dim,
                        self.feature_dim,
                        mean=random_weight["mean"],
                        std=random_weight["std"],
                    )
                    self.last_fc = RandomWeightFactorization(
                        self.feature_dim,
                        final_output_dim,
                        mean=random_weight["mean"],
                        std=random_weight["std"],
                    )
                else:
                    self.feature_proj = nn.Linear(body_feature_dim, self.feature_dim)
                    self.last_fc = nn.Linear(self.feature_dim, final_output_dim)

                self.feature_act = _build_activation(bottleneck_activation, self.feature_dim)

            def body_features_tensor(self, x):
                u = self.embed_u(x)
                v = self.embed_v(x)
                y = x
                for block in self.blocks:
                    y = block(y, u, v)
                return y

            def hidden_features_tensor(self, x):
                y = self.body_features_tensor(x)
                y = self.feature_proj(y)
                if self.feature_act is not None:
                    y = self.feature_act(y)
                return y

            def forward_tensor(self, x):
                return self.last_fc(self.hidden_features_tensor(x))

        return _BottleneckPirateNet


def build_bottleneck_piratenet(**kwargs):
    """Build a bottleneck PirateNet without importing Paddle at module import time."""
    return BottleneckPirateNet.build_class()(**kwargs)


def load_matching_pretrained(model, path: str, *, verbose: bool = True):
    """Load only matching parameters from a Paddle checkpoint."""
    import paddle

    checkpoint_path = str(path)
    if checkpoint_path.endswith(".pdparams"):
        load_path = checkpoint_path
    else:
        load_path = f"{checkpoint_path}.pdparams"

    state_dict = paddle.load(load_path)
    current_state = model.state_dict()
    merged_state = dict(current_state)
    matched_keys = []
    skipped_keys = []

    for key, value in state_dict.items():
        if key not in current_state:
            skipped_keys.append((key, "missing_in_model"))
            continue
        if list(current_state[key].shape) != list(value.shape):
            skipped_keys.append((key, f"shape_mismatch {list(value.shape)} != {list(current_state[key].shape)}"))
            continue
        merged_state[key] = value
        matched_keys.append(key)

    missing_keys, unexpected_keys = model.set_state_dict(merged_state)

    if verbose:
        print(
            "Loaded partial pretrained weights: "
            f"{len(matched_keys)} matched, {len(skipped_keys)} skipped, "
            f"{len(missing_keys)} missing_after_set, {len(unexpected_keys)} unexpected_after_set"
        )
        if skipped_keys:
            preview = ", ".join(f"{key}({reason})" for key, reason in skipped_keys[:8])
            suffix = " ..." if len(skipped_keys) > 8 else ""
            print(f"  Skipped keys: {preview}{suffix}")

    return {
        "matched_keys": matched_keys,
        "skipped_keys": skipped_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }
