#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.solver import (
    PseudoTimePDESolver,
)
from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.utils.config import (
    PseudoTimePDEConfig,
)


def _to_builtin(value: Any):
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _summarize_lsq_steps(step_stats):
    if not step_stats:
        return {}

    residual_norms = [float(step["residual_norm"]) for step in step_stats]
    relative_residuals = [float(step["relative_residual"]) for step in step_stats]
    condition_numbers = [
        float(step["condition_number"])
        for step in step_stats
        if step.get("condition_number") is not None
    ]
    solve_times = [float(step["solve_time"]) for step in step_stats]

    return {
        "num_steps": len(step_stats),
        "residual_norm_min": min(residual_norms),
        "residual_norm_max": max(residual_norms),
        "relative_residual_min": min(relative_residuals),
        "relative_residual_max": max(relative_residuals),
        "condition_number_min": min(condition_numbers) if condition_numbers else None,
        "condition_number_max": max(condition_numbers) if condition_numbers else None,
        "solve_time_total": sum(solve_times),
        "solve_time_avg": sum(solve_times) / len(solve_times),
    }


def _write_text_summary(path: Path, payload: dict) -> None:
    convergence = payload["convergence_info"]
    lsq_summary = payload["lsq_summary"]
    lines = [
        f"case_dir: {payload['case_dir']}",
        f"device: {payload['device']}",
        f"linear_device: {payload['linear_device']}",
        f"one_shot_spotter: {payload['one_shot_spotter']}",
        f"spotter_skip: {payload['spotter_skip']}",
        f"training_loss_last: {payload['training_loss_last']}",
        "",
        "[convergence]",
        f"converged: {convergence.get('converged')}",
        f"iterations: {convergence.get('iterations')}",
        f"final_residual: {convergence.get('final_residual')}",
        f"elapsed_time: {convergence.get('elapsed_time')}",
        f"reason: {convergence.get('reason', '')}",
        "",
        "[lsq_summary]",
        f"num_steps: {lsq_summary.get('num_steps')}",
        f"residual_norm_min: {lsq_summary.get('residual_norm_min')}",
        f"residual_norm_max: {lsq_summary.get('residual_norm_max')}",
        f"relative_residual_min: {lsq_summary.get('relative_residual_min')}",
        f"relative_residual_max: {lsq_summary.get('relative_residual_max')}",
        f"condition_number_min: {lsq_summary.get('condition_number_min')}",
        f"condition_number_max: {lsq_summary.get('condition_number_max')}",
        f"solve_time_total: {lsq_summary.get('solve_time_total')}",
        f"solve_time_avg: {lsq_summary.get('solve_time_avg')}",
        "",
        "[steps]",
    ]

    for step in payload["lsq_steps"]:
        lines.append(
            "step={step} method={method} shape={matrix_shape} cond={condition_number} "
            "res2={residual_norm:.6e} rel={relative_residual:.6e} rank={rank} "
            "solve_time={solve_time:.3f}".format(**step)
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a pseudo-time case and export least-squares diagnostics for each Snipper step."
    )
    parser.add_argument(
        "--case-dir",
        required=True,
        help="Path to the pseudo-time case directory.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path. Defaults to <case>/results/lsq_trace.json",
    )
    parser.add_argument(
        "--output-txt",
        default=None,
        help="Optional text summary output path. Defaults to <case>/results/lsq_trace.txt",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override network device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--linear-device",
        default="cpu",
        help="Override linear solver device. Default is cpu.",
    )
    parser.add_argument(
        "--one-shot-spotter",
        action="store_true",
        help="Train Spotter only once at iter=0, then let Snipper finish pseudo-time stepping.",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    config = PseudoTimePDEConfig(case_dir=str(case_dir))
    config.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config.linear_device = args.linear_device
    if args.one_shot_spotter:
        config.spotter_skip = int(config.max_iterations) + 1

    solver = PseudoTimePDESolver(config=config)
    results = solver.solve()

    lsq_steps = [_to_builtin(step) for step in solver.fitter.solver.current_step_stats]
    payload = {
        "case_dir": str(case_dir),
        "device": config.device,
        "linear_device": config.linear_device,
        "one_shot_spotter": bool(args.one_shot_spotter),
        "spotter_skip": int(config.spotter_skip),
        "training_loss_last": _to_builtin(getattr(solver, "_last_training_loss", None)),
        "convergence_info": _to_builtin(results["convergence_info"]),
        "lsq_summary": _summarize_lsq_steps(lsq_steps),
        "lsq_steps": lsq_steps,
    }

    results_dir = case_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else results_dir / "lsq_trace.json"
    output_txt = Path(args.output_txt) if args.output_txt else results_dir / "lsq_trace.txt"

    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_text_summary(output_txt, payload)

    print(f"Saved JSON trace to: {output_json}")
    print(f"Saved text trace to: {output_txt}")
    print(f"One-shot Spotter: {payload['one_shot_spotter']} (spotter_skip={payload['spotter_skip']})")
    for step in payload["lsq_steps"]:
        print(
            f"step={step['step']:>2} "
            f"res2={step['residual_norm']:.6e} "
            f"rel={step['relative_residual']:.6e} "
            f"cond={step['condition_number']:.6e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
