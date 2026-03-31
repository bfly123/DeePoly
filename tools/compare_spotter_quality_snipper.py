#!/usr/bin/env python3
"""Compare whether coarse spotters benefit more from snipper correction."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-case",
        default="cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D_paddlesci_bottleneck_snipper_10x6_8k_p55_gpu",
        help="Base case directory relative to repo root.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="tmp/paddle_bottleneck_h256_b20_adam/checkpoints",
        help="Directory containing Paddle checkpoint prefixes / .pdparams files.",
    )
    parser.add_argument(
        "--checkpoints",
        default="epoch_10,epoch_50,epoch_100,best_model",
        help="Comma-separated checkpoint names inside checkpoint-dir. Suffix .pdparams is optional.",
    )
    parser.add_argument(
        "--variant-root",
        default="tmp/spotter_quality_compare",
        help="Directory for generated comparison cases, relative to repo root.",
    )
    parser.add_argument("--iterations", type=int, default=3, help="Fixed-point max iterations.")
    parser.add_argument("--under-relaxation", type=float, default=0.2, help="Snipper under-relaxation omega.")
    parser.add_argument("--device", default="cuda", help="Value for config.device.")
    parser.add_argument("--linear-device", default="cuda", help="Value for config.linear_device.")
    parser.add_argument("--spotter-device", default="gpu", help="Value for spotter_feature.factory_kwargs.device.")
    parser.add_argument("--linear-method", default="gpu_direct", help="Value for config.linear_method.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for `python -m src.cli`.")
    parser.add_argument("--skip-autocode", action="store_true", help="Pass --skip-autocode to src.cli.")
    parser.add_argument("--keep-old-results", action="store_true", help="Keep existing case dirs instead of recreating them.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_checkpoint_name(name: str) -> str:
    item = name.strip()
    if not item:
        raise ValueError("empty checkpoint name")
    return item[:-9] if item.endswith(".pdparams") else item


def absolutize_case_paths(cfg: dict[str, Any], base_case_dir: Path) -> None:
    ref = cfg.get("reference_solution")
    if isinstance(ref, str) and ref:
        cfg["reference_solution"] = str((base_case_dir / ref).resolve())
    feature_cfg = cfg.get("spotter_feature")
    if not isinstance(feature_cfg, dict):
        return
    factory_file = feature_cfg.get("factory_file")
    if isinstance(factory_file, str) and factory_file:
        feature_cfg["factory_file"] = str((base_case_dir / factory_file).resolve())
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
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    absolutize_case_paths(cfg, base_case_dir)
    cfg["device"] = args.device
    cfg["linear_device"] = args.linear_device
    cfg["linear_method"] = args.linear_method
    cfg["auto_code"] = False

    fix_cfg = dict(cfg.get("fixiteration", {}) or {})
    fix_cfg["max_iterations"] = int(args.iterations)
    fix_cfg["under_relaxation"] = float(args.under_relaxation)
    cfg["fixiteration"] = fix_cfg

    feature_cfg = dict(cfg.get("spotter_feature", {}) or {})
    factory_kwargs = dict(feature_cfg.get("factory_kwargs", {}) or {})
    factory_kwargs["device"] = args.spotter_device
    factory_kwargs["pretrained_model_path"] = str(checkpoint_path.resolve())
    feature_cfg["factory_kwargs"] = factory_kwargs
    cfg["spotter_feature"] = feature_cfg
    return cfg


def prepare_variant_case(
    base_case_dir: Path,
    variant_root: Path,
    checkpoint_name: str,
    cfg: dict[str, Any],
    keep_old_results: bool,
) -> Path:
    case_dir = variant_root / f"{base_case_dir.name}_{checkpoint_name}"
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


def _load_reference(case_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
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
    return np.column_stack([X_ref.ravel(), Y_ref.ravel()]), u_ref.ravel()


def _compute_rel_l2_from_solution_data(case_dir: Path, key: str) -> float | None:
    solution_path = case_dir / "results" / "solution_data.npz"
    if not solution_path.exists():
        return None
    reference = _load_reference(case_dir)
    if reference is None:
        return None
    ref_points, ref_values = reference
    data = np.load(solution_path, allow_pickle=True)
    if key not in data:
        return None
    coords = np.asarray(data["x_test"], dtype=np.float64)
    values = np.asarray(data[key], dtype=np.float64).reshape(-1)
    pred = griddata(coords, values, ref_points, method="linear")
    if pred is None:
        return None
    if np.isnan(pred).any():
        nearest = griddata(coords, values, ref_points[np.isnan(pred)], method="nearest")
        pred[np.isnan(pred)] = nearest
    err = pred - ref_values
    denom = float(np.linalg.norm(ref_values)) + 1e-12
    return float(np.linalg.norm(err) / denom)


def summarize_case(
    case_dir: Path,
    checkpoint_name: str,
    elapsed_seconds: float,
    return_code: int,
) -> dict[str, Any]:
    report_path = case_dir / "results" / "accuracy_report.json"
    report = load_json(report_path) if report_path.exists() else {}
    spotter = dict(report.get("spotter", {}) or {})
    snipper = dict(report.get("snipper_direct", {}) or {})
    fixed_point = dict(report.get("fixed_point", {}) or {})

    if "rel_l2" not in spotter:
        rel = _compute_rel_l2_from_solution_data(case_dir, "spotter_test")
        if rel is not None:
            spotter["rel_l2"] = rel
    if "rel_l2" not in snipper:
        rel = _compute_rel_l2_from_solution_data(case_dir, "sniper_test")
        if rel is not None:
            snipper["rel_l2"] = rel
        else:
            solution_path = case_dir / "results" / "solution_data.npz"
            if solution_path.exists():
                with np.load(solution_path, allow_pickle=True) as data:
                    if "U_test" in data:
                        rel = _compute_rel_l2_from_solution_data(case_dir, "U_test")
                        if rel is not None:
                            snipper["rel_l2"] = rel

    spotter_rel = spotter.get("rel_l2")
    snipper_rel = snipper.get("rel_l2")
    abs_gain = None
    rel_gain_pct = None
    if spotter_rel is not None and snipper_rel is not None:
        abs_gain = float(spotter_rel - snipper_rel)
        rel_gain_pct = float(abs_gain / (spotter_rel + 1e-12) * 100.0)

    return {
        "checkpoint": checkpoint_name,
        "return_code": return_code,
        "elapsed_seconds": elapsed_seconds,
        "spotter_rel_l2": spotter_rel,
        "snipper_rel_l2": snipper_rel,
        "abs_gain": abs_gain,
        "rel_gain_pct": rel_gain_pct,
        "fixed_point_iterations": fixed_point.get("iterations"),
        "fixed_point_final_residual": fixed_point.get("final_residual"),
        "case_dir": str(case_dir),
    }


def write_summary(rows: list[dict[str, Any]], summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    fieldnames = [
        "checkpoint",
        "return_code",
        "elapsed_seconds",
        "spotter_rel_l2",
        "snipper_rel_l2",
        "abs_gain",
        "rel_gain_pct",
        "fixed_point_iterations",
        "fixed_point_final_residual",
        "case_dir",
    ]
    with open(summary_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = repo_root()
    base_case_dir = (root / args.base_case).resolve()
    checkpoint_dir = (root / args.checkpoint_dir).resolve()
    variant_root = (root / args.variant_root).resolve()
    base_cfg = load_json(base_case_dir / "config.json")
    checkpoint_names = [normalize_checkpoint_name(item) for item in args.checkpoints.split(",") if item.strip()]

    rows: list[dict[str, Any]] = []
    failures = 0
    for checkpoint_name in checkpoint_names:
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pdparams"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        cfg = build_variant_config(base_cfg, base_case_dir, checkpoint_path, args)
        case_dir = prepare_variant_case(base_case_dir, variant_root, checkpoint_name, cfg, args.keep_old_results)
        start = time.time()
        result = run_case(case_dir, args.python, args.skip_autocode)
        elapsed_seconds = time.time() - start
        write_text(case_dir / "run_stdout.log", result.stdout)
        write_text(case_dir / "run_stderr.log", result.stderr)
        row = summarize_case(case_dir, checkpoint_name, elapsed_seconds, result.returncode)
        rows.append(row)
        print(
            f"[{checkpoint_name}] return_code={result.returncode} "
            f"spotter_rel_l2={row['spotter_rel_l2']} "
            f"snipper_rel_l2={row['snipper_rel_l2']} "
            f"gain={row['abs_gain']} residual={row['fixed_point_final_residual']}"
        )
        if result.returncode != 0:
            failures += 1

    summary_dir = variant_root / "summary"
    write_summary(rows, summary_dir)
    print(f"summary json: {summary_dir / 'summary.json'}")
    print(f"summary csv : {summary_dir / 'summary.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
