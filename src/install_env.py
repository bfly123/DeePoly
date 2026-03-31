#!/usr/bin/env python3
"""Helpers for validating the current Python environment before CLI install."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


MIN_PYTHON = (3, 10)
REQUIRED_RUNTIME_MODULES = (
    "numpy",
    "scipy",
    "matplotlib",
    "sympy",
    "torch",
)


@dataclass(frozen=True)
class EnvironmentCheckResult:
    python_executable: str
    python_version: str
    python_ok: bool
    env_name: str | None
    env_prefix: str
    scripts_dir: str
    missing_modules: tuple[str, ...]
    torch_cuda_available: bool | None


def detect_environment_name(env: dict[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env
    conda_env = env.get("CONDA_DEFAULT_ENV")
    if conda_env:
        return conda_env

    virtual_env = env.get("VIRTUAL_ENV")
    if virtual_env:
        return Path(virtual_env).name

    return None


def is_supported_python(version_info: tuple[int, ...] | None = None) -> bool:
    if version_info is None:
        version_info = sys.version_info
    return tuple(version_info[:2]) >= MIN_PYTHON


def find_missing_modules(
    required_modules: Iterable[str] = REQUIRED_RUNTIME_MODULES,
    importer: Callable[[str], object] = importlib.import_module,
) -> tuple[str, ...]:
    missing: list[str] = []
    for module_name in required_modules:
        try:
            importer(module_name)
        except Exception:
            missing.append(module_name)
    return tuple(missing)


def collect_environment_check(
    importer: Callable[[str], object] = importlib.import_module,
    env: dict[str, str] | None = None,
) -> EnvironmentCheckResult:
    missing_modules = find_missing_modules(importer=importer)
    torch_cuda_available = None
    if "torch" not in missing_modules:
        torch_module = importer("torch")
        torch_cuda_available = bool(torch_module.cuda.is_available())

    return EnvironmentCheckResult(
        python_executable=sys.executable,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        python_ok=is_supported_python(),
        env_name=detect_environment_name(env=env),
        env_prefix=sys.prefix,
        scripts_dir=sysconfig.get_path("scripts"),
        missing_modules=missing_modules,
        torch_cuda_available=torch_cuda_available,
    )


def format_environment_report(result: EnvironmentCheckResult) -> str:
    env_name = result.env_name or "<none>"
    cuda_status = "unavailable"
    if result.torch_cuda_available is True:
        cuda_status = "available"
    elif result.torch_cuda_available is False:
        cuda_status = "not available"

    lines = [
        f"Python executable: {result.python_executable}",
        f"Python version: {result.python_version}",
        f"Active environment: {env_name}",
        f"Environment prefix: {result.env_prefix}",
        f"Scripts directory: {result.scripts_dir}",
        f"PyTorch CUDA status: {cuda_status}",
    ]

    if result.missing_modules:
        lines.append(
            "Missing runtime modules: " + ", ".join(result.missing_modules)
        )
    else:
        lines.append("Missing runtime modules: <none>")

    return "\n".join(lines)


def validate_environment(result: EnvironmentCheckResult) -> list[str]:
    errors: list[str] = []
    if not result.python_ok:
        errors.append(
            "Python >= 3.10 is required for DeePoly installation in the current environment."
        )
    if result.missing_modules:
        errors.append(
            "Current environment is missing runtime dependencies: "
            + ", ".join(result.missing_modules)
            + ". Install them into this environment before running install.sh."
        )
    return errors


def console_script_path(script_name: str = "deepoly") -> Path:
    return Path(sysconfig.get_path("scripts")) / script_name


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.install_env",
        description="Validate the current environment for DeePoly CLI installation.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="check",
        choices=["check", "script-path"],
        help="Validation command to execute.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "script-path":
        print(console_script_path())
        return 0

    result = collect_environment_check()
    print(format_environment_report(result))
    errors = validate_environment(result)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
