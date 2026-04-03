#!/usr/bin/env python3
"""Sweep Allen-Cahn pseudo-time cases and report global error statistics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

from src.abstract_class.config.config_io import load_config_file, resolve_case_config_path, save_config_file


def parse_pair_list(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.lower().split("x", 1)
        pairs.append((int(left), int(right)))
    if not pairs:
        raise ValueError("expected at least one AxB pair")
    return pairs


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def parse_domain_specs(raw: str) -> list[int | tuple[int, int]]:
    specs: list[int | tuple[int, int]] = []
    for item in raw.split(","):
        value = item.strip().lower()
        if not value:
            continue
        if "x" in value:
            left, right = value.split("x", 1)
            specs.append((int(left), int(right)))
        else:
            specs.append(int(value))
    if not specs:
        raise ValueError("expected at least one domain-point specification")
    return specs


def format_domain_spec(value: int | tuple[int, int]) -> str:
    if isinstance(value, tuple):
        return f"{value[0]}x{value[1]}"
    return str(value)


def materialize_point_count(value: int | tuple[int, int]) -> int:
    if isinstance(value, tuple):
        return int(value[0] * value[1])
    return int(value)


@dataclass
class RunSpec:
    n_segments: tuple[int, int]
    poly_degree: tuple[int, int]
    points_domain: int | tuple[int, int]
    epochs_adam: int
    points_domain_test: tuple[int, int]

    @property
    def label(self) -> str:
        seg = f"s{self.n_segments[0]}x{self.n_segments[1]}"
        poly = f"p{self.poly_degree[0]}x{self.poly_degree[1]}"
        pts = f"d{format_domain_spec(self.points_domain)}"
        ep = f"e{self.epochs_adam}"
        test = f"g{self.points_domain_test[0]}x{self.points_domain_test[1]}"
        return "_".join((seg, poly, pts, ep, test))


def build_run_specs(args: argparse.Namespace) -> list[RunSpec]:
    segment_pairs = parse_pair_list(args.segments)
    poly_pairs = parse_pair_list(args.poly_degrees)
    domain_points = parse_domain_specs(args.points_domain)
    epoch_values = parse_int_list(args.epochs_adam)
    test_pairs = parse_pair_list(args.points_domain_test)
    return [
        RunSpec(
            n_segments=segments,
            poly_degree=poly,
            points_domain=points,
            epochs_adam=epochs,
            points_domain_test=test_grid,
        )
        for segments, poly, points, epochs, test_grid in itertools.product(
            segment_pairs,
            poly_pairs,
            domain_points,
            epoch_values,
            test_pairs,
        )
    ]


def make_variant_config(base_config: dict[str, Any], spec: RunSpec) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_config))
    cfg["n_segments"] = list(spec.n_segments)
    cfg["poly_degree"] = list(spec.poly_degree)
    cfg["points_domain"] = list(spec.points_domain) if isinstance(spec.points_domain, tuple) else int(spec.points_domain)
    cfg["points_domain_test"] = list(spec.points_domain_test)
    spotter = dict(cfg.get("spotter", {}))
    spotter["epochs_adam"] = int(spec.epochs_adam)
    cfg["spotter"] = spotter
    cfg["auto_code"] = False
    return cfg


def prepare_case_dir(base_case: Path, case_root: Path, spec: RunSpec, base_config: dict[str, Any]) -> Path:
    case_dir = case_root / spec.label
    case_dir.mkdir(parents=True, exist_ok=True)
    ref_link = case_dir / "reference_data"
    if ref_link.exists() or ref_link.is_symlink():
        if ref_link.is_symlink() or ref_link.is_file():
            ref_link.unlink()
    if not ref_link.exists():
        ref_link.symlink_to(base_case / "reference_data", target_is_directory=True)
    cfg = make_variant_config(base_config, spec)
    save_config_file(case_dir / "config.json", cfg)
    return case_dir


def load_reference(reference_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = loadmat(reference_path)
    x_ref = np.asarray(mat["x"]).reshape(-1)
    t_ref = np.asarray(mat["t"]).reshape(-1)
    u_ref = np.asarray(mat["usol"], dtype=np.float64)
    return x_ref, t_ref, u_ref


def _prediction_grid(coords: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])
    grid = np.full((len(x_unique), len(y_unique)), np.nan, dtype=np.float64)
    x_index = {value: idx for idx, value in enumerate(x_unique)}
    y_index = {value: idx for idx, value in enumerate(y_unique)}
    for (x_val, y_val), u_val in zip(coords, values, strict=False):
        grid[x_index[float(x_val)], y_index[float(y_val)]] = float(u_val)
    if np.isnan(grid).any():
        raise ValueError("prediction grid contains NaNs after remapping")
    return x_unique, y_unique, grid


def compute_metrics_from_results(results_dir: Path, x_ref: np.ndarray, t_ref: np.ndarray, u_ref: np.ndarray) -> dict[str, Any]:
    solution_data = np.load(results_dir / "solution_data.npz", allow_pickle=True)
    coords = np.asarray(solution_data["x_test"], dtype=np.float64)
    spotter_values = np.asarray(solution_data["spotter_test"], dtype=np.float64).reshape(-1)
    sniper_values = np.asarray(solution_data["sniper_test"], dtype=np.float64).reshape(-1)
    x_unique, t_unique, spotter_grid = _prediction_grid(coords, spotter_values)
    _, _, sniper_grid = _prediction_grid(coords, sniper_values)

    x_mesh_ref, t_mesh_ref = np.meshgrid(x_ref, t_ref, indexing="xy")
    ref_points = np.column_stack([x_mesh_ref.ravel(), t_mesh_ref.ravel()])

    results: dict[str, Any] = {
        "evaluation_grid": {
            "reference_x": int(len(x_ref)),
            "reference_t": int(len(t_ref)),
            "numerical_x": int(len(x_unique)),
            "numerical_t": int(len(t_unique)),
            "numerical_x_min": float(x_unique.min()),
            "numerical_x_max": float(x_unique.max()),
            "numerical_t_min": float(t_unique.min()),
            "numerical_t_max": float(t_unique.max()),
        }
    }
    ref_norm = float(np.linalg.norm(u_ref.ravel()))

    for name, grid in (("spotter", spotter_grid), ("sniper", sniper_grid)):
        interpolator = RegularGridInterpolator((x_unique, t_unique), grid, bounds_error=False, fill_value=None)
        pred = interpolator(ref_points).reshape(len(t_ref), len(x_ref))
        err = pred - u_ref
        abs_err = np.abs(err)
        results[name] = {
            "l1_mean_abs": float(np.mean(abs_err)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "linf_abs": float(np.max(abs_err)),
            "rel_l2": float(np.linalg.norm(err.ravel()) / ref_norm),
            "mean_prediction": float(np.mean(pred)),
            "min_prediction": float(np.min(pred)),
            "max_prediction": float(np.max(pred)),
        }
    return results


def run_case(case_dir: Path, results_dir: Path, device: str, linear_device: str | None, skip_autocode: bool) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "src.cli", str(case_dir), "--device", device, "--results-dir", str(results_dir)]
    if linear_device:
        cmd.extend(["--linear-device", linear_device])
    if skip_autocode:
        cmd.append("--skip-autocode")
    return subprocess.run(
        cmd,
        cwd=repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "success",
        "n_segments",
        "poly_degree",
        "points_domain",
        "points_domain_count",
        "epochs_adam",
        "points_domain_test",
        "elapsed_seconds",
        "spotter_rel_l2",
        "sniper_rel_l2",
        "spotter_rmse",
        "sniper_rmse",
        "spotter_linf_abs",
        "sniper_linf_abs",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics", {})
            writer.writerow(
                {
                    "label": row["label"],
                    "success": row["success"],
                    "n_segments": "x".join(map(str, row["n_segments"])),
                    "poly_degree": "x".join(map(str, row["poly_degree"])),
                    "points_domain": format_domain_spec(row["points_domain"]),
                    "points_domain_count": materialize_point_count(row["points_domain"]),
                    "epochs_adam": row["epochs_adam"],
                    "points_domain_test": "x".join(map(str, row["points_domain_test"])),
                    "elapsed_seconds": f"{row.get('elapsed_seconds', math.nan):.3f}",
                    "spotter_rel_l2": metrics.get("spotter", {}).get("rel_l2"),
                    "sniper_rel_l2": metrics.get("sniper", {}).get("rel_l2"),
                    "spotter_rmse": metrics.get("spotter", {}).get("rmse"),
                    "sniper_rmse": metrics.get("sniper", {}).get("rmse"),
                    "spotter_linf_abs": metrics.get("spotter", {}).get("linf_abs"),
                    "sniper_linf_abs": metrics.get("sniper", {}).get("linf_abs"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep AC_equation case and compute global error metrics.")
    parser.add_argument("--case", default="cases/nonlinear_pde_cases/pseudotime_pde_cases/AC_equation")
    parser.add_argument("--segments", default="2x2,3x3,4x4")
    parser.add_argument("--poly-degrees", default="7x5,9x6")
    parser.add_argument("--points-domain", default="6000,10000")
    parser.add_argument("--epochs-adam", default="6000,10000")
    parser.add_argument("--points-domain-test", default="100x100")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--linear-device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    case_dir = (root / args.case).resolve()
    config_path = Path(resolve_case_config_path(case_dir))

    results_root = Path(args.results_root).resolve() if args.results_root else (case_dir / "sweeps")
    if args.clean and results_root.exists():
        shutil.rmtree(results_root)
    generated_case_root = results_root / "generated_cases"
    run_root = results_root / "runs"
    generated_case_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    base_config = load_config_file(config_path)
    x_ref, t_ref, u_ref = load_reference(case_dir / base_config["reference_solution"])

    run_specs = build_run_specs(args)
    if args.max_runs > 0:
        run_specs = run_specs[: args.max_runs]

    summary_rows: list[dict[str, Any]] = []
    skip_autocode = False

    print(f"Sweeping {len(run_specs)} AC-equation configurations")
    for index, spec in enumerate(run_specs, start=1):
        print(f"[{index}/{len(run_specs)}] {spec.label}")
        variant_case_dir = prepare_case_dir(case_dir, generated_case_root, spec, base_config)
        variant_results_dir = run_root / spec.label
        variant_results_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()
        completed = run_case(
            case_dir=variant_case_dir,
            results_dir=variant_results_dir,
            device=args.device,
            linear_device=args.linear_device,
            skip_autocode=skip_autocode,
        )
        elapsed = time.perf_counter() - start_time
        save_text(variant_results_dir / "stdout.log", completed.stdout)
        save_text(variant_results_dir / "stderr.log", completed.stderr)

        row: dict[str, Any] = {
            "label": spec.label,
            "n_segments": list(spec.n_segments),
            "poly_degree": list(spec.poly_degree),
            "points_domain": spec.points_domain,
            "epochs_adam": spec.epochs_adam,
            "points_domain_test": list(spec.points_domain_test),
            "elapsed_seconds": elapsed,
            "success": completed.returncode == 0,
        }

        if completed.returncode == 0:
            try:
                metrics = compute_metrics_from_results(variant_results_dir, x_ref, t_ref, u_ref)
                row["metrics"] = metrics
                with open(variant_results_dir / "metrics.json", "w", encoding="utf-8") as handle:
                    json.dump(metrics, handle, indent=2, ensure_ascii=False)
                print(
                    "  sniper relL2={:.4e}, rmse={:.4e}, linf={:.4e}".format(
                        metrics["sniper"]["rel_l2"],
                        metrics["sniper"]["rmse"],
                        metrics["sniper"]["linf_abs"],
                    )
                )
            except Exception as exc:  # pragma: no cover - batch robustness
                row["success"] = False
                row["error"] = f"metric computation failed: {exc}"
                print(f"  metric computation failed: {exc}")
        else:
            row["error"] = f"solver failed with exit code {completed.returncode}"
            print(f"  solver failed with exit code {completed.returncode}")

        summary_rows.append(row)
        skip_autocode = True

    best_by_grid: dict[str, dict[str, Any]] = {}
    successful_rows = [row for row in summary_rows if row.get("success") and "metrics" in row]
    for row in successful_rows:
        key = "x".join(map(str, row["n_segments"]))
        current = best_by_grid.get(key)
        if current is None or row["metrics"]["sniper"]["rel_l2"] < current["metrics"]["sniper"]["rel_l2"]:
            best_by_grid[key] = row

    summary = {
        "case": str(case_dir),
        "results_root": str(results_root),
        "total_runs": len(summary_rows),
        "successful_runs": len(successful_rows),
        "runs": summary_rows,
        "best_by_grid": best_by_grid,
    }
    with open(results_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_summary_csv(summary_rows, results_root / "summary.csv")
    with open(results_root / "best_by_grid.json", "w", encoding="utf-8") as handle:
        json.dump(best_by_grid, handle, indent=2, ensure_ascii=False)

    print("\nBest configuration by segmentation grid:")
    for key in sorted(best_by_grid):
        row = best_by_grid[key]
        metrics = row["metrics"]["sniper"]
        print(
            "  grid {} -> {} | relL2={:.4e}, rmse={:.4e}, linf={:.4e}".format(
                key,
                row["label"],
                metrics["rel_l2"],
                metrics["rmse"],
                metrics["linf_abs"],
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
