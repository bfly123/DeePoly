#!/usr/bin/env python3
"""Installed DeePoly command line interface."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict

from src.meta_coding.auto_code_manager import AutoCodeManager


SOLVER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "time_pde": {
        "config_module": "src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils",
        "config_class": "TimePDEConfig",
        "solver_module": "src.problem_solvers.nonlinear_pde_solver.time_pde_solver",
        "solver_class": "TimePDESolver",
    },
    "func_fitting": {
        "config_module": "src.problem_solvers.func_fitting_solver.utils",
        "config_class": "FuncFittingConfig",
        "solver_module": "src.problem_solvers.func_fitting_solver.solver",
        "solver_class": "FuncFittingSolver",
    },
    "linear_pde": {
        "config_module": "src.problem_solvers.linear_pde_solver.utils",
        "config_class": "LinearPDEConfig",
        "solver_module": "src.problem_solvers.linear_pde_solver",
        "solver_class": "LinearPDESolver",
    },
    "general_pde": {
        "config_module": "src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.utils",
        "config_class": "PseudoTimePDEConfig",
        "solver_module": "src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver",
        "solver_class": "PseudoTimePDESolver",
    },
    "fixiteration_pde": {
        "config_module": "src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.utils",
        "config_class": "FixIterationPDEConfig",
        "solver_module": "src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver",
        "solver_class": "FixIterationPDESolver",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deepoly",
        description="Run a DeePoly case directory.",
    )
    parser.add_argument(
        "case_path",
        nargs="?",
        default=".",
        help="Case directory or config.json path. Default: current directory.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Override config.device. Default: auto.",
    )
    parser.add_argument(
        "--linear-device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override config.linear_device.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override results output directory.",
    )
    parser.add_argument(
        "--skip-autocode",
        action="store_true",
        help="Skip automatic code generation / consistency checks.",
    )
    return parser.parse_args(argv)


def resolve_case_dir(case_path: str) -> str:
    path = Path(case_path).expanduser().resolve()
    if path.is_file():
        if path.name != "config.json":
            raise FileNotFoundError(f"Expected a case directory or config.json, got: {path}")
        path = path.parent

    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return str(path)


def load_problem_type(case_dir: str) -> str:
    config_path = Path(case_dir) / "config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        config_dict = json.load(handle)
    problem_type = config_dict.get("problem_type")
    if not problem_type:
        raise ValueError(f"Missing 'problem_type' field in {config_path}")
    return problem_type


def resolve_device(device: str | None) -> str | None:
    if device is None:
        return None
    if device == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def import_symbol(module_name: str, symbol_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def maybe_handle_autocode(case_dir: str, skip_autocode: bool) -> bool:
    if skip_autocode:
        try:
            manager = AutoCodeManager(case_dir)
            need_regenerate, reason = manager.check_consistency()
            if need_regenerate:
                print(
                    "Warning: --skip-autocode is set while generated PDE code is inconsistent "
                    f"with this case ({reason}). The run may use stale operator code."
                )
        except Exception as exc:
            print(f"Warning: failed to verify auto code consistency: {exc}")
        return False
    manager = AutoCodeManager(case_dir)
    return manager.handle_autocode_workflow()


def create_solver(problem_type: str, case_dir: str, args: argparse.Namespace):
    solver_info = SOLVER_REGISTRY.get(problem_type)
    if solver_info is None:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    if "config_module" in solver_info:
        config_class = import_symbol(solver_info["config_module"], solver_info["config_class"])
    else:
        config_class = solver_info["config_class"]

    if "solver_module" in solver_info:
        solver_class = import_symbol(solver_info["solver_module"], solver_info["solver_class"])
    else:
        solver_class = solver_info["solver_class"]
    config = config_class(case_dir=case_dir)

    device = resolve_device(args.device)
    if device is not None:
        config.device = device

    linear_device = resolve_device(args.linear_device)
    if linear_device is not None and hasattr(config, "linear_device"):
        config.linear_device = linear_device

    if args.results_dir:
        config.results_dir = os.path.abspath(args.results_dir)
        os.makedirs(config.results_dir, exist_ok=True)

    return solver_class(config=config, case_dir=case_dir)


def run_case(args: argparse.Namespace) -> None:
    case_dir = resolve_case_dir(args.case_path)
    print(f"Using case directory: {case_dir}")

    if maybe_handle_autocode(case_dir, args.skip_autocode):
        return

    problem_type = load_problem_type(case_dir)
    print(f"Problem type: {problem_type}")

    solver = create_solver(problem_type, case_dir, args)
    solver.solve()
    print("Solving completed successfully!")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
