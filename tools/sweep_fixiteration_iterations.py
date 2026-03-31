#!/usr/bin/env python3
"""Sweep fixed-point iteration counts for a DeePoly fixiteration case."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import griddata
from scipy.io import loadmat


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_iterations(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one iteration count")
    for value in values:
        if value <= 0:
            raise ValueError("iteration counts must be positive")
    return values


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one float value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-case",
        default="cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D_paddlesci_bottleneck_snipper_10x6_8k_p55_gpu",
        help="Base case directory relative to repo root.",
    )
    parser.add_argument(
        "--iterations",
        default="1,3,5",
        help="Comma-separated fixed-point iteration counts to test.",
    )
    parser.add_argument(
        "--under-relaxations",
        default="1.0",
        help="Comma-separated under-relaxation factors omega to test.",
    )
    parser.add_argument(
        "--variant-root",
        default="tmp/fixiter_iteration_sweep",
        help="Directory for generated variant cases, relative to repo root.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Value for top-level config.device.",
    )
    parser.add_argument(
        "--linear-device",
        default="cuda",
        help="Value for top-level config.linear_device.",
    )
    parser.add_argument(
        "--spotter-device",
        default="gpu",
        help="Value for spotter_feature.factory_kwargs.device if present.",
    )
    parser.add_argument(
        "--linear-method",
        default="",
        help="Override config.linear_method. Leave empty to preserve the base case setting.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch `python -m src.cli`.",
    )
    parser.add_argument(
        "--skip-autocode",
        action="store_true",
        help="Pass --skip-autocode to src.cli.",
    )
    parser.add_argument(
        "--keep-old-results",
        action="store_true",
        help="Keep existing generated variant directories instead of recreating them.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def absolutize_case_paths(cfg: dict[str, Any], base_case_dir: Path) -> None:
    reference_solution = cfg.get("reference_solution")
    if isinstance(reference_solution, str) and reference_solution:
        cfg["reference_solution"] = str((base_case_dir / reference_solution).resolve())

    feature_cfg = cfg.get("spotter_feature")
    if not isinstance(feature_cfg, dict):
        return

    for key in ("factory_file",):
        value = feature_cfg.get(key)
        if isinstance(value, str) and value:
            feature_cfg[key] = str((base_case_dir / value).resolve())

    factory_kwargs = feature_cfg.get("factory_kwargs")
    if not isinstance(factory_kwargs, dict):
        return

    for key in ("paddlesci_root", "pretrained_model_path"):
        value = factory_kwargs.get(key)
        if isinstance(value, str) and value:
            factory_kwargs[key] = str((base_case_dir / value).resolve())


def build_variant_config(
    base_cfg: dict[str, Any],
    base_case_dir: Path,
    iteration_count: int,
    under_relaxation: float,
    device: str,
    linear_device: str,
    spotter_device: str,
    linear_method: str,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    absolutize_case_paths(cfg, base_case_dir)

    cfg["device"] = device
    cfg["linear_device"] = linear_device
    cfg["auto_code"] = False

    fix_cfg = dict(cfg.get("fixiteration", {}) or {})
    fix_cfg["max_iterations"] = int(iteration_count)
    fix_cfg["under_relaxation"] = float(under_relaxation)
    cfg["fixiteration"] = fix_cfg

    if linear_method:
        cfg["linear_method"] = linear_method

    feature_cfg = cfg.get("spotter_feature")
    if isinstance(feature_cfg, dict):
        factory_kwargs = dict(feature_cfg.get("factory_kwargs", {}) or {})
        if factory_kwargs:
            factory_kwargs["device"] = spotter_device
            feature_cfg["factory_kwargs"] = factory_kwargs
        cfg["spotter_feature"] = feature_cfg

    return cfg


def prepare_variant_case(
    base_case_dir: Path,
    variant_root: Path,
    iteration_count: int,
    under_relaxation: float,
    cfg: dict[str, Any],
    keep_old_results: bool,
) -> Path:
    omega_label = str(under_relaxation).replace(".", "p")
    case_dir = variant_root / f"{base_case_dir.name}_iter{iteration_count}_omega{omega_label}"
    if case_dir.exists() and not keep_old_results:
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    dump_json(case_dir / "config.json", cfg)
    return case_dir


def run_case(case_dir: Path, python_exe: str, skip_autocode: bool) -> subprocess.CompletedProcess[str]:
    cmd = [python_exe, "-m", "src.cli", str(case_dir)]
    if skip_autocode:
        cmd.append("--skip-autocode")
    return subprocess.run(
        cmd,
        cwd=repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def summarize_row(
    case_dir: Path,
    iteration_count: int,
    elapsed_seconds: float,
    return_code: int,
) -> dict[str, Any]:
    report_path = case_dir / "results" / "accuracy_report.json"
    report = load_json(report_path) if report_path.exists() else {}

    spotter = dict(report.get("spotter", {}) or {})
    snipper = dict(report.get("snipper_direct", {}) or {})
    fixed_point = dict(report.get("fixed_point", {}) or {})
    snipper_system = dict(report.get("snipper_system", {}) or {})
    if not snipper:
        snipper = _fallback_metrics_from_solution_data(case_dir) or {}

    return {
        "case_dir": str(case_dir),
        "iteration_count": iteration_count,
        "return_code": return_code,
        "elapsed_seconds": elapsed_seconds,
        "spotter_rel_l2": spotter.get("rel_l2"),
        "snipper_rel_l2": snipper.get("rel_l2"),
        "spotter_rmse": spotter.get("rmse"),
        "snipper_rmse": snipper.get("rmse"),
        "spotter_max_abs": spotter.get("max_abs"),
        "snipper_max_abs": snipper.get("max_abs"),
        "fixed_point_iterations_done": fixed_point.get("iterations"),
        "fixed_point_final_residual": fixed_point.get("final_residual"),
        "linear_method": snipper_system.get("linear_method"),
        "feature_dtype": snipper_system.get("feature_dtype"),
    }


def _load_reference_for_case(case_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    cfg = load_json(case_dir / "config.json")
    ref_path = cfg.get("reference_solution")
    if not ref_path:
        return None
    ref_file = Path(ref_path)
    if not ref_file.is_absolute():
        ref_file = (case_dir / ref_file).resolve()
    if not ref_file.exists():
        return None
    mat = loadmat(ref_file)
    x_ref = np.asarray(mat["x"]).reshape(-1)
    y_ref = np.asarray(mat.get("t", mat.get("y"))).reshape(-1)
    u_ref = np.asarray(mat.get("usol", mat.get("u")), dtype=np.float64)
    X_ref, Y_ref = np.meshgrid(x_ref, y_ref, indexing="xy")
    ref_points = np.column_stack([X_ref.ravel(), Y_ref.ravel()])
    ref_values = u_ref.ravel()
    return ref_points, ref_values


def _fallback_metrics_from_solution_data(case_dir: Path) -> dict[str, float] | None:
    solution_path = case_dir / "results" / "solution_data.npz"
    if not solution_path.exists():
        return None
    reference = _load_reference_for_case(case_dir)
    if reference is None:
        return None
    ref_points, ref_values = reference
    ref_norm = float(np.linalg.norm(ref_values)) + 1e-12

    data = np.load(solution_path, allow_pickle=True)
    coords = np.asarray(data["x_test"], dtype=np.float64)
    values = np.asarray(data["U_test"], dtype=np.float64).reshape(-1)

    pred_on_ref = griddata(coords, values, ref_points, method="linear")
    if pred_on_ref is None:
        return None
    if np.isnan(pred_on_ref).any():
        nearest = griddata(coords, values, ref_points[np.isnan(pred_on_ref)], method="nearest")
        pred_on_ref[np.isnan(pred_on_ref)] = nearest

    err = pred_on_ref - ref_values
    return {
        "rel_l2": float(np.linalg.norm(err) / ref_norm),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
    }


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "under_relaxation",
        "iteration_count",
        "return_code",
        "elapsed_seconds",
        "spotter_rel_l2",
        "snipper_rel_l2",
        "spotter_rmse",
        "snipper_rmse",
        "spotter_max_abs",
        "snipper_max_abs",
        "fixed_point_iterations_done",
        "fixed_point_final_residual",
        "linear_method",
        "feature_dtype",
        "case_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    root = repo_root()
    base_case_dir = (root / args.base_case).resolve()
    if not (base_case_dir / "config.json").exists():
        raise FileNotFoundError(f"case config not found: {base_case_dir / 'config.json'}")

    variant_root = (root / args.variant_root).resolve()
    iterations = parse_iterations(args.iterations)
    under_relaxations = parse_float_list(args.under_relaxations)
    base_cfg = load_json(base_case_dir / "config.json")

    rows: list[dict[str, Any]] = []
    failures = 0

    for under_relaxation in under_relaxations:
        for iteration_count in iterations:
            cfg = build_variant_config(
                base_cfg=base_cfg,
                base_case_dir=base_case_dir,
                iteration_count=iteration_count,
                under_relaxation=under_relaxation,
                device=args.device,
                linear_device=args.linear_device,
                spotter_device=args.spotter_device,
                linear_method=args.linear_method,
            )
            case_dir = prepare_variant_case(
                base_case_dir=base_case_dir,
                variant_root=variant_root,
                iteration_count=iteration_count,
                under_relaxation=under_relaxation,
                cfg=cfg,
                keep_old_results=args.keep_old_results,
            )

            start = time.time()
            result = run_case(case_dir, args.python, args.skip_autocode)
            elapsed_seconds = time.time() - start

            write_text(case_dir / "run_stdout.log", result.stdout)
            write_text(case_dir / "run_stderr.log", result.stderr)

            row = summarize_row(
                case_dir=case_dir,
                iteration_count=iteration_count,
                elapsed_seconds=elapsed_seconds,
                return_code=result.returncode,
            )
            row["under_relaxation"] = float(under_relaxation)
            rows.append(row)

            print(
                f"[iter={iteration_count}, omega={under_relaxation}] return_code={result.returncode} "
                f"spotter_rel_l2={row['spotter_rel_l2']} "
                f"snipper_rel_l2={row['snipper_rel_l2']} "
                f"residual={row['fixed_point_final_residual']}"
            )

            if result.returncode != 0:
                failures += 1

    summary_dir = variant_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    dump_json(summary_dir / "summary.json", {"rows": rows})
    write_summary_csv(rows, summary_dir / "summary.csv")

    print(f"summary json: {summary_dir / 'summary.json'}")
    print(f"summary csv : {summary_dir / 'summary.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
