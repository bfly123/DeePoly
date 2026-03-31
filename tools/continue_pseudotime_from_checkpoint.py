#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path

import numpy as np
import torch

from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.solver import (
    PseudoTimePDESolver,
)
from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.utils import (
    PseudoTimePDEConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue a pseudo-time PDE result from a saved checkpoint directory.",
    )
    parser.add_argument("checkpoint_dir", type=Path, help="Existing results directory to continue from.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Output directory for the continued run.")
    parser.add_argument("--case-dir", type=Path, default=None, help="Case directory. Defaults to checkpoint_dir parent.")
    parser.add_argument("--steps", type=int, required=True, help="Number of additional pseudo-time steps to run.")
    parser.add_argument(
        "--tau-init",
        type=float,
        default=None,
        help="Initial tau for continuation. Defaults to returned_tau/final_tau from convergence_info.json when available.",
    )
    parser.add_argument("--dt", type=float, default=None, help="Pseudo-time step to use for continuation.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Override config.device.",
    )
    parser.add_argument(
        "--linear-device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override config.linear_device.",
    )
    parser.add_argument(
        "--points-domain-test",
        nargs=2,
        type=int,
        metavar=("NX", "NY"),
        default=None,
        help="Override test grid resolution.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip Spotter retraining and only continue Snipper pseudo-time steps.",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help="Optional reference solution path for regression report generation.",
    )
    return parser.parse_args()


def resolve_device(device: str | None) -> str | None:
    if device is None:
        return None
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_checkpoint_state(checkpoint_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    solution_path = checkpoint_dir / "solution_data.npz"
    if not solution_path.exists():
        raise FileNotFoundError(f"Missing solution data: {solution_path}")
    data = np.load(solution_path)
    return data["sniper_train"], data["coeffs"]


def load_tau_init(checkpoint_dir: Path, tau_init: float | None) -> float:
    if tau_init is not None:
        return float(tau_init)

    info_path = checkpoint_dir / "convergence_info.json"
    if not info_path.exists():
        return 0.0

    info = json.loads(info_path.read_text(encoding="utf-8"))
    if "returned_tau" in info:
        return float(info["returned_tau"])
    if "final_tau" in info:
        return float(info["final_tau"])
    return 0.0


def load_model_weights(model_path: Path, device: str):
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def build_regression_report_if_requested(results_dir: Path, reference_path: Path | None) -> None:
    if reference_path is None:
        return
    from tools.compare_lid_driven_cavity_regression import build_regression_report

    report = build_regression_report(results_dir / "solution_data.npz", reference_path)
    report_path = results_dir / "regression_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved regression report to: {report_path}")


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    case_dir = (args.case_dir or checkpoint_dir.parent).expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    U_init, coeffs_init = load_checkpoint_state(checkpoint_dir)
    tau_init = load_tau_init(checkpoint_dir, args.tau_init)

    config = PseudoTimePDEConfig(case_dir=str(case_dir))
    config.max_iterations = int(args.steps)
    config.results_dir = str(results_dir)
    config.realtime_visualization = False
    config.save_animation = False
    config.visualization.realtime_visualization = False
    config.visualization.save_animation = False

    device = resolve_device(args.device)
    if device is not None:
        config.device = device
    linear_device = resolve_device(args.linear_device)
    if linear_device is not None:
        config.linear_device = linear_device
    if args.points_domain_test is not None:
        config.points_domain_test = list(args.points_domain_test)
    if args.dt is not None:
        dt_value = float(args.dt)
        config.dt_init = dt_value
        config.dt_max = max(config.dt_max, dt_value)
        config.dt_min = min(config.dt_min, dt_value)
        config.rollback_dt_min = min(getattr(config, "rollback_dt_min", config.dt_min), dt_value)

    solver = PseudoTimePDESolver(config=config, case_dir=str(case_dir))
    model_path = checkpoint_dir / "spotter_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    solver.model.load_state_dict(load_model_weights(model_path, config.device))
    solver.model.to(config.device)
    solver.model.eval()
    solver.fitter.fitter_init(solver.model)

    if args.skip_training:
        solver._should_train_network = types.MethodType(lambda self, iteration: False, solver)

    start_time = time.time()
    dt = float(args.dt) if args.dt is not None else float(config.dt_init)
    U_final, coeffs_final, conv_info = solver._solve_iterations(
        U_init.copy(),
        dt,
        tau_init,
        coeffs_init.copy(),
        start_time,
    )

    U_sniper_test = solver._interpolate_to_test_grid(solver._to_numpy_cpu(U_final))
    U_spotter = solver._get_network_prediction()
    U_spotter_test = solver._interpolate_to_test_grid(solver._to_numpy_cpu(U_spotter))
    results = {
        "solution": U_final,
        "coefficients": coeffs_final,
        "convergence_info": conv_info,
        "config": solver.config,
        "predictions": {"sniper": U_sniper_test, "spotter": U_spotter_test},
        "predictions_train": {
            "sniper": solver._to_numpy_cpu(U_final),
            "spotter": solver._to_numpy_cpu(U_spotter),
        },
        "flow_diagnostics": solver._build_flow_diagnostic_results(coeffs_final),
    }
    solver.visualizer.render(results, solver.data_train, solver.data_test, solver.reference_solution)
    solver._save_spotter_model_checkpoint(solver.config.results_dir)
    build_regression_report_if_requested(results_dir, args.reference_path)
    print(f"Continuation finished: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
