#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT = (
    PROJECT_ROOT
    / "cases"
    / "nonlinear_pde_cases"
    / "pseudotime_pde_cases"
    / "lid_driven_cavity_re100"
    / "results"
    / "solution_data.npz"
)
DEFAULT_REFERENCE = (
    Path("/home/bfly/workspace/computeforcfd/混合网络/2D_NS_Stable") / "u_nRe100.npy"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "cases"
    / "nonlinear_pde_cases"
    / "pseudotime_pde_cases"
    / "lid_driven_cavity_re100"
    / "results"
    / "regression_report.json"
)


def _series_stats(values: np.ndarray) -> dict:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _quantiles(values: np.ndarray) -> dict:
    q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
    return {"q05": float(q05), "q50": float(q50), "q95": float(q95)}


def _compare_component(reference: np.ndarray, current: np.ndarray, *, center: bool) -> dict:
    ref = reference.copy()
    cur = current.copy()
    if center:
        ref = ref - np.mean(ref)
        cur = cur - np.mean(cur)

    ref_q = _quantiles(ref)
    cur_q = _quantiles(cur)
    return {
        "reference_stats": _series_stats(ref),
        "current_stats": _series_stats(cur),
        "reference_quantiles": ref_q,
        "current_quantiles": cur_q,
        "quantile_abs_diff": {
            key: abs(cur_q[key] - ref_q[key]) for key in ref_q
        },
    }


def build_regression_report(
    current_path: Path = DEFAULT_CURRENT,
    reference_path: Path = DEFAULT_REFERENCE,
) -> dict:
    if not current_path.exists():
        raise FileNotFoundError(f"Current solution file not found: {current_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference solution file not found: {reference_path}")

    reference = np.load(reference_path)
    current_npz = np.load(current_path)

    report = {
        "reference_path": str(reference_path),
        "current_path": str(current_path),
        "reference_shape": list(reference.shape),
        "current_shapes": {},
        "comparisons": {},
    }

    for key in current_npz.files:
        if key.endswith("_train") or key.endswith("_test"):
            report["current_shapes"][key] = list(current_npz[key].shape)

    for current_key in ["spotter_test", "sniper_test"]:
        if current_key not in current_npz:
            continue
        current = current_npz[current_key]
        current_report = {}
        for idx, var_name in enumerate(["u", "v", "p"]):
            current_report[var_name] = _compare_component(
                reference[:, idx],
                current[:, idx],
                center=(var_name == "p"),
            )
        report["comparisons"][current_key] = current_report

    return report


def main() -> int:
    report = build_regression_report()
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved regression report to: {DEFAULT_REPORT}")
    for source_name, source_report in report["comparisons"].items():
        print(f"\n[{source_name}]")
        for var_name, metrics in source_report.items():
            diffs = metrics["quantile_abs_diff"]
            print(
                f"  {var_name}: "
                f"q05={diffs['q05']:.6e}, "
                f"q50={diffs['q50']:.6e}, "
                f"q95={diffs['q95']:.6e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
