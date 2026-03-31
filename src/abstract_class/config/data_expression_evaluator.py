from typing import Iterable

import numpy as np


class MathExpressionEvaluator:
    """Evaluate config math expressions against point clouds."""

    def __init__(self, spatial_vars: Iterable[str]):
        self.spatial_vars = list(spatial_vars)
        self.namespace = {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "tanh": np.tanh,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "exp": np.exp,
            "pi": np.pi,
            "sqrt": np.sqrt,
            "log": np.log,
            "log10": np.log10,
            "abs": np.abs,
            "np": np,
        }

    def evaluate(self, expr: str, x: np.ndarray) -> np.ndarray:
        func_str = f"lambda {', '.join(self.spatial_vars)}: {expr}"
        try:
            value_func = eval(func_str, self.namespace)
            result = value_func(*[x[:, i] for i in range(x.shape[1])])
            if isinstance(result, (int, float)):
                return np.full((x.shape[0], 1), result)
            return np.asarray(result).reshape(-1, 1)
        except Exception:
            return np.zeros((x.shape[0], 1))

    def evaluate_piecewise(self, value_dict: dict, x: np.ndarray) -> np.ndarray:
        if value_dict.get("type") != "piecewise":
            return np.zeros((x.shape[0], 1))
        var = value_dict.get("variable")
        if var not in self.spatial_vars:
            return np.zeros((x.shape[0], 1))
        var_idx = self.spatial_vars.index(var)
        x_var = x[:, var_idx]
        result = np.zeros((x.shape[0], 1))
        processed = np.zeros((x.shape[0],), dtype=bool)
        for piece in value_dict.get("pieces", []):
            range_min, range_max = piece.get("range", [float("-inf"), float("inf")])
            expression = piece.get("expression", "0")
            mask = (x_var >= range_min) & (x_var < range_max) & ~processed
            if isinstance(expression, str):
                result[mask] = self.evaluate(expression, x)[mask]
            else:
                result[mask] = expression
            processed[mask] = True
        return result


def normalize_column_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.shape[1] != 1:
        return values[:, 0:1]
    return values
