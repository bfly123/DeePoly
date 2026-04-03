from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.spotter_runtime.config_support import extract_provider_options


def build_func_fitting_context(config, data_train: Dict[str, Any], provider_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if data_train is None:
        raise ValueError("func_fitting backend requires training data")
    x_train = np.asarray(data_train.get("x"), dtype=np.float64)
    y_train = np.asarray(data_train.get("u"), dtype=np.float64)
    if x_train.ndim != 2:
        raise ValueError(f"func_fitting expects x with shape (N, dim), got {x_train.shape}")
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_train.ndim != 2:
        raise ValueError(f"func_fitting expects u with shape (N, neq), got {y_train.shape}")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("func_fitting x/u sample counts do not match")
    provider_options = extract_provider_options(provider_cfg)
    n_dim = int(getattr(config, 'n_dim', x_train.shape[1]))
    n_eqs = int(getattr(config, 'n_eqs', y_train.shape[1]))
    if x_train.shape[1] != n_dim:
        raise ValueError(f"func_fitting dimension mismatch: x has {x_train.shape[1]}, config has {n_dim}")
    if y_train.shape[1] != n_eqs:
        raise ValueError(f"func_fitting output mismatch: u has {y_train.shape[1]}, config has {n_eqs}")
    return {
        'program_kind': 'func_fitting',
        'x_train': x_train,
        'y_train': y_train,
        'n_dim': n_dim,
        'n_eqs': n_eqs,
        'hidden_dims': tuple(int(value) for value in getattr(config, 'hidden_dims', (32, 32))),
        'learning_rate': float(provider_options.get('learning_rate', getattr(config, 'learning_rate', 1e-3))),
        'epochs_adam': int(provider_options.get('epochs_adam', getattr(config, 'epochs_adam', 0))),
        'max_retries': int(provider_options.get('max_retries', getattr(config, 'max_retries', 0))),
        'dnn_tol': float(provider_options.get('dnn_tol', provider_options.get('DNNtol', getattr(config, 'DNNtol', 1e-6)))),
        'seed': int(provider_options.get('seed', getattr(config, 'seed', 42))),
        'backend_options': provider_options,
    }


def supports_func_fitting_builtin(config, data_train: Dict[str, Any] | None = None) -> bool:
    try:
        sample = data_train if data_train is not None else {'x': np.zeros((1, max(int(getattr(config, 'n_dim', 1)), 1))), 'u': np.zeros((1, max(int(getattr(config, 'n_eqs', 1)), 1)))}
        build_func_fitting_context(config, sample)
    except Exception:
        return False
    return str(getattr(config, 'problem_type', '')).lower() == 'func_fitting'
