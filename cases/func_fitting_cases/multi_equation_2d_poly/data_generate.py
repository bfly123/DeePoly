import numpy as np


def _u_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        1.0
        + 0.5 * x
        - 0.25 * y
        + 0.75 * x * y
        + 0.10 * x**2
        - 0.20 * y**2
        + 0.05 * x**3
    )


def _v_field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        -0.3
        + 0.4 * x
        + 0.6 * y
        - 0.5 * x * y
        + 0.2 * x**2
        + 0.3 * y**2
        + 0.1 * x**2 * y
        - 0.08 * x * y**2
    )


def generate_global_field(x: np.ndarray) -> np.ndarray:
    x_coord = x[:, 0]
    y_coord = x[:, 1]
    u = _u_field(x_coord, y_coord)
    v = _v_field(x_coord, y_coord)
    return np.column_stack((u, v))


def generate_reference_solution(x: np.ndarray) -> np.ndarray:
    return generate_global_field(x)

