#!/usr/bin/env python3
"""Evaluate solution_data.npz against the case reference solution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import griddata
from scipy.io import loadmat

from src.abstract_class.config.config_io import load_config_file, resolve_case_config_path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict[str, Any]:
    return load_config_file(path)


def resolve_reference(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    cfg = load_json(Path(resolve_case_config_path(case_dir)))
    ref_path = cfg.get("reference_solution")
    if not ref_path:
        raise FileNotFoundError("config.reference_solution is missing")
    ref_file = Path(ref_path)
    if not ref_file.is_absolute():
        ref_file = (case_dir / ref_file).resolve()
    mat = loadmat(ref_file)
    x_ref = np.asarray(mat["x"]).reshape(-1)
    y_ref = np.asarray(mat.get("t", mat.get("y"))).reshape(-1)
    u_ref = np.asarray(mat.get("usol", mat.get("u")), dtype=np.float64)
    X_ref, Y_ref = np.meshgrid(x_ref, y_ref, indexing="xy")
    return np.column_stack([X_ref.ravel(), Y_ref.ravel()]), u_ref.ravel()


def compute_rel_l2(coords: np.ndarray, values: np.ndarray, ref_points: np.ndarray, ref_values: np.ndarray) -> dict[str, float]:
    pred = griddata(coords, values, ref_points, method="linear")
    if pred is None:
        raise RuntimeError("griddata(linear) returned None")
    if np.isnan(pred).any():
        nearest = griddata(coords, values, ref_points[np.isnan(pred)], method="nearest")
        pred[np.isnan(pred)] = nearest
    err = pred - ref_values
    denom = float(np.linalg.norm(ref_values)) + 1e-12
    return {
        "rel_l2": float(np.linalg.norm(err) / denom),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", help="Case directory containing config.yaml/config.yml/config.json and results/solution_data.npz")
    parser.add_argument(
        "--keys",
        default="spotter_test,sniper_test,U_test",
        help="Comma-separated candidate arrays in solution_data.npz to evaluate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_dir = Path(args.case_dir).expanduser().resolve()
    solution_path = case_dir / "results" / "solution_data.npz"
    if not solution_path.exists():
        raise FileNotFoundError(f"solution data not found: {solution_path}")
    payload = np.load(solution_path, allow_pickle=True)
    coords = np.asarray(payload["x_test"], dtype=np.float64)
    ref_points, ref_values = resolve_reference(case_dir)

    results: dict[str, Any] = {"case_dir": str(case_dir), "metrics": {}}
    for key in [item.strip() for item in args.keys.split(",") if item.strip()]:
        if key not in payload:
            continue
        values = np.asarray(payload[key], dtype=np.float64).reshape(-1)
        results["metrics"][key] = compute_rel_l2(coords, values, ref_points, ref_values)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
