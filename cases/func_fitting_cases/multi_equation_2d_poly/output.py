import json
import os

import matplotlib.pyplot as plt
import numpy as np

from data_generate import generate_reference_solution


def _grid_from_points(x_coords, values, decimals=12):
    rounded_points = np.round(np.asarray(x_coords, dtype=np.float64), decimals=decimals)
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    unique_points, inverse = np.unique(rounded_points, axis=0, return_inverse=True)
    if unique_points.size == 0:
        return None, None, None

    reduced_values = np.zeros(unique_points.shape[0], dtype=np.float64)
    counts = np.zeros(unique_points.shape[0], dtype=np.int32)
    np.add.at(reduced_values, inverse, values)
    np.add.at(counts, inverse, 1)
    reduced_values /= np.maximum(counts, 1)

    x_unique = np.unique(unique_points[:, 0])
    y_unique = np.unique(unique_points[:, 1])
    if x_unique.size * y_unique.size != unique_points.shape[0]:
        return None, None, None

    grid = np.full((y_unique.size, x_unique.size), np.nan, dtype=np.float64)
    x_index = {float(x): idx for idx, x in enumerate(x_unique)}
    y_index = {float(y): idx for idx, y in enumerate(y_unique)}
    for point, value in zip(unique_points, reduced_values):
        grid[y_index[float(point[1])], x_index[float(point[0])]] = value
    if np.isnan(grid).any():
        return None, None, None

    return x_unique, y_unique, grid


def _save_component_plot(x_coords, truth, pred, var_name, save_path):
    x_unique, y_unique, truth_grid = _grid_from_points(x_coords, truth)
    _, _, pred_grid = _grid_from_points(x_coords, pred)
    _, _, err_grid = _grid_from_points(x_coords, pred - truth)
    if truth_grid is None or pred_grid is None or err_grid is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    images = [
        (truth_grid, f"{var_name} Truth"),
        (pred_grid, f"{var_name} Prediction"),
        (err_grid, f"{var_name} Error"),
    ]
    extent = [x_unique.min(), x_unique.max(), y_unique.min(), y_unique.max()]
    for ax, (grid, title) in zip(axes, images):
        image = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        fig.colorbar(image, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_output(
    config,
    data_train,
    data_test,
    fitter,
    model,
    coeffs,
    result_dir,
    visualizer,
    total_time,
    scoper_time,
    sniper_time,
    **kwargs,
):
    del fitter, model, visualizer
    os.makedirs(result_dir, exist_ok=True)

    train_predictions = np.asarray(kwargs["train_predictions"], dtype=np.float64)
    test_predictions = np.asarray(kwargs["test_predictions"], dtype=np.float64)
    train_truth = np.asarray(data_train["u"], dtype=np.float64)
    test_truth = generate_reference_solution(np.asarray(data_test["x"], dtype=np.float64))

    train_error = train_predictions - train_truth
    test_error = test_predictions - test_truth

    metrics = {
        "timing": {
            "total_time": float(total_time),
            "scoper_time": float(scoper_time),
            "sniper_time": float(sniper_time),
        },
        "variables": {},
    }
    for eq_idx, var_name in enumerate(config.vars_list):
        train_abs = np.abs(train_error[:, eq_idx])
        test_abs = np.abs(test_error[:, eq_idx])
        metrics["variables"][var_name] = {
            "train_linf": float(np.max(train_abs)),
            "train_rmse": float(np.sqrt(np.mean(train_error[:, eq_idx] ** 2))),
            "test_linf": float(np.max(test_abs)),
            "test_rmse": float(np.sqrt(np.mean(test_error[:, eq_idx] ** 2))),
        }
        _save_component_plot(
            np.asarray(data_test["x"], dtype=np.float64),
            test_truth[:, eq_idx],
            test_predictions[:, eq_idx],
            var_name,
            os.path.join(result_dir, f"{var_name}_comparison.png"),
        )

    with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, ensure_ascii=False)

    np.savez(
        os.path.join(result_dir, "solution_data.npz"),
        x_train=np.asarray(data_train["x"], dtype=np.float64),
        x_test=np.asarray(data_test["x"], dtype=np.float64),
        train_truth=train_truth,
        train_predictions=train_predictions,
        train_error=train_error,
        test_truth=test_truth,
        test_predictions=test_predictions,
        test_error=test_error,
        coeffs=np.asarray(coeffs, dtype=np.float64),
    )

    print(f"Saved multi-equation function-fitting outputs to: {result_dir}")
