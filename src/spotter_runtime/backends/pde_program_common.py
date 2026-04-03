from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np

from src.meta_coding import parse_operators
from src.spotter_runtime.config_support import extract_provider_options


def _as_float64_array(
    value: Any,
    *,
    ndim: int | None = None,
    empty_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if value is None:
        if empty_shape is None:
            return np.array([], dtype=np.float64)
        return np.zeros(empty_shape, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 and empty_shape is not None:
        return np.zeros(empty_shape, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected array with ndim={ndim}, got shape={array.shape}")
    return array


def _normalize_standard_boundary_data(
    bc_data: Mapping[str, Any],
    *,
    n_dim: int,
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "x": _as_float64_array(bc_data.get("x"), ndim=2, empty_shape=(0, n_dim)),
        "values": _as_float64_array(bc_data.get("values"), ndim=2, empty_shape=(0, 1)),
        "weights": _as_float64_array(bc_data.get("weights"), ndim=2, empty_shape=(0, 1)),
    }
    if "normals" in bc_data:
        normalized["normals"] = _as_float64_array(
            bc_data.get("normals"),
            ndim=2,
            empty_shape=(0, n_dim),
        )
    if "params" in bc_data:
        normalized["params"] = bc_data.get("params")
    return normalized


def _normalize_periodic_boundary_data(
    bc_data: Mapping[str, Any],
    *,
    n_dim: int,
) -> Dict[str, Any]:
    pairs = []
    for pair in bc_data.get("pairs", []) or []:
        pairs.append(
            {
                "x_1": _as_float64_array(pair.get("x_1"), ndim=2, empty_shape=(0, n_dim)),
                "x_2": _as_float64_array(pair.get("x_2"), ndim=2, empty_shape=(0, n_dim)),
                "weight": float(pair.get("weight", 1.0)),
            }
        )
    return {"pairs": pairs}


def normalize_boundary_program(
    global_boundary_dict: Mapping[Any, Any],
    *,
    n_dim: int,
) -> Dict[int, Dict[str, Any]]:
    normalized: Dict[int, Dict[str, Any]] = {}
    for raw_var_idx, raw_entry in (global_boundary_dict or {}).items():
        var_idx = int(raw_var_idx)
        entry = dict(raw_entry or {})
        normalized[var_idx] = {
            "dirichlet": _normalize_standard_boundary_data(entry.get("dirichlet", {}), n_dim=n_dim),
            "neumann": _normalize_standard_boundary_data(entry.get("neumann", {}), n_dim=n_dim),
            "robin": _normalize_standard_boundary_data(entry.get("robin", {}), n_dim=n_dim),
            "periodic": _normalize_periodic_boundary_data(entry.get("periodic", {}), n_dim=n_dim),
        }
    return normalized


def _extract_operator_parse(config) -> Dict[str, Any]:
    parsed = dict(getattr(config, "operator_parse", {}) or {})
    if parsed:
        return parsed
    return parse_operators(
        getattr(config, "eq", {}) or {},
        list(getattr(config, "vars_list", []) or []),
        list(getattr(config, "spatial_vars", []) or []),
        getattr(config, "constants", {}) or getattr(config, "const_list", {}) or {},
    )


def _build_symbol_info(operator_parse: Mapping[str, Any], derivatives: list[list[int]]) -> Dict[str, Dict[str, Any]]:
    symbol_info: Dict[str, Dict[str, Any]] = {}
    for name, (var_idx, deriv_idx) in (operator_parse.get("all_derivatives", {}) or {}).items():
        symbol_info[str(name)] = {
            "var_idx": int(var_idx),
            "derivative_pattern": tuple(int(order) for order in derivatives[int(deriv_idx)]),
        }
    return symbol_info


def _build_symbolic_pde_context(
    config,
    data_train: Dict[str, Any],
    provider_cfg: Dict[str, Any] | None,
    *,
    problem_type: str,
    residual_mode: str,
) -> Dict[str, Any]:
    if data_train is None:
        raise ValueError(f"{problem_type} backend requires training data")
    if str(getattr(config, "problem_type", "")).lower() != str(problem_type).lower():
        raise ValueError(f"builtin backend only supports problem_type={problem_type}")

    x_train = _as_float64_array(data_train.get("x"), ndim=2)
    n_dim = int(getattr(config, "n_dim", x_train.shape[1]))
    if x_train.shape[1] != n_dim:
        raise ValueError(f"{problem_type} dimension mismatch: x has {x_train.shape[1]}, config has {n_dim}")

    provider_options = extract_provider_options(provider_cfg)
    operator_parse = _extract_operator_parse(config)
    derivatives = [
        [int(order) for order in pattern]
        for pattern in (operator_parse.get("derivatives", []) or [])
    ]
    constants = dict(getattr(config, "constants", {}) or {})
    constants.setdefault("pi", float(np.pi))

    return {
        "program_kind": f"{problem_type}_generic",
        "problem_type": str(problem_type),
        "residual_mode": str(residual_mode),
        "time_scheme": getattr(config, "time_scheme", None),
        "x_train": x_train,
        "boundary_program": normalize_boundary_program(
            data_train.get("global_boundary_dict", {}) or {},
            n_dim=n_dim,
        ),
        "n_dim": n_dim,
        "n_eqs": int(getattr(config, "n_eqs", 1)),
        "vars_list": tuple(str(value) for value in getattr(config, "vars_list", ()) or ()),
        "spatial_vars": tuple(str(value) for value in getattr(config, "spatial_vars", ()) or ()),
        "hidden_dims": tuple(int(value) for value in getattr(config, "hidden_dims", (32, 32))),
        "learning_rate": float(provider_options.get("learning_rate", getattr(config, "learning_rate", 1e-3))),
        "epochs_adam": int(provider_options.get("epochs_adam", getattr(config, "epochs_adam", 0))),
        "max_retries": int(provider_options.get("max_retries", getattr(config, "max_retries", 0))),
        "dnn_tol": float(
            provider_options.get(
                "dnn_tol",
                provider_options.get("DNNtol", getattr(config, "DNNtol", 1e-6)),
            )
        ),
        "boundary_constraint_weight": float(
            provider_options.get(
                "boundary_constraint_weight",
                getattr(config, "boundary_constraint_weight", 10.0),
            )
        ),
        "seed": int(provider_options.get("seed", getattr(config, "seed", 42))),
        "constants": constants,
        "derivatives": tuple(tuple(pattern) for pattern in derivatives),
        "symbol_info": _build_symbol_info(operator_parse, derivatives),
        "operator_terms": {
            key: list(operator_parse.get("operator_terms", {}).get(key, []) or [])
            for key in ("L1", "L2", "F", "N", "S")
        },
        "backend_options": provider_options,
    }


def build_linear_pde_context(
    config,
    data_train: Dict[str, Any],
    provider_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return _build_symbolic_pde_context(
        config,
        data_train,
        provider_cfg,
        problem_type="linear_pde",
        residual_mode="linear_pde",
    )


def build_time_pde_context(
    config,
    data_train: Dict[str, Any],
    provider_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return _build_symbolic_pde_context(
        config,
        data_train,
        provider_cfg,
        problem_type="time_pde",
        residual_mode="time_pde",
    )


def build_symbolic_pde_context(
    config,
    data_train: Dict[str, Any],
    provider_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    problem_type = str(getattr(config, "problem_type", "") or "").lower()
    if problem_type == "linear_pde":
        return build_linear_pde_context(config, data_train, provider_cfg=provider_cfg)
    if problem_type == "time_pde":
        return build_time_pde_context(config, data_train, provider_cfg=provider_cfg)
    raise ValueError(f"Unsupported symbolic PDE problem_type: {problem_type}")


def supports_linear_pde_builtin(
    config,
    data_train: Dict[str, Any] | None = None,
) -> bool:
    try:
        sample = data_train
        if sample is None:
            sample = {
                "x": np.zeros((1, max(int(getattr(config, "n_dim", 1)), 1)), dtype=np.float64),
                "global_boundary_dict": {},
            }
        build_linear_pde_context(config, sample)
    except Exception:
        return False
    return True


def supports_time_pde_builtin(
    config,
    data_train: Dict[str, Any] | None = None,
) -> bool:
    try:
        sample = data_train
        if sample is None:
            sample = {
                "x": np.zeros((1, max(int(getattr(config, "n_dim", 1)), 1)), dtype=np.float64),
                "global_boundary_dict": {},
            }
        build_time_pde_context(config, sample)
    except Exception:
        return False
    return True


def supports_symbolic_pde_builtin(
    config,
    data_train: Dict[str, Any] | None = None,
) -> bool:
    try:
        sample = data_train
        if sample is None:
            sample = {
                "x": np.zeros((1, max(int(getattr(config, "n_dim", 1)), 1)), dtype=np.float64),
                "global_boundary_dict": {},
            }
        build_symbolic_pde_context(config, sample)
    except Exception:
        return False
    return True


def iter_all_boundary_arrays(
    boundary_program: Mapping[int, Mapping[str, Any]],
) -> Iterable[np.ndarray]:
    for entry in (boundary_program or {}).values():
        for bc_type in ("dirichlet", "neumann", "robin"):
            x = np.asarray(entry.get(bc_type, {}).get("x", []), dtype=np.float64)
            if x.size > 0:
                yield x
        for pair in entry.get("periodic", {}).get("pairs", []) or []:
            x_1 = np.asarray(pair.get("x_1", []), dtype=np.float64)
            x_2 = np.asarray(pair.get("x_2", []), dtype=np.float64)
            if x_1.size > 0:
                yield x_1
            if x_2.size > 0:
                yield x_2
