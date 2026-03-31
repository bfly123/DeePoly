import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_src_package():
    result = run_python("import src")
    assert result.returncode == 0, result.stderr


def test_import_visualization_package():
    result = run_python("import src.visualization")
    assert result.returncode == 0, result.stderr


def test_import_problem_solvers_package():
    result = run_python("import src.problem_solvers")
    assert result.returncode == 0, result.stderr


def test_import_meta_coding_package():
    result = run_python("import src.meta_coding")
    assert result.returncode == 0, result.stderr


def test_import_linear_solver_symbol():
    result = run_python("from src.algebraic_solver import LinearSolver")
    assert result.returncode == 0, result.stderr


def test_import_fastnewton_symbol_without_cupy():
    result = run_python("from src.algebraic_solver import FastNewtonSolver")
    assert result.returncode == 0, result.stderr
