"""Lazy exports for problem solver packages."""

from importlib import import_module

_EXPORTS = {
    "func_fitting_solver": ("src.problem_solvers", "func_fitting_solver"),
    "linear_pde_solver": ("src.problem_solvers", "linear_pde_solver"),
    "nonlinear_pde_solver": ("src.problem_solvers", "nonlinear_pde_solver"),
    "FuncFittingSolver": ("src.problem_solvers.func_fitting_solver.solver", "FuncFittingSolver"),
    "LinearPDESolver": ("src.problem_solvers.linear_pde_solver", "LinearPDESolver"),
    "NonlinearPDESolver": ("src.problem_solvers.nonlinear_pde_solver", "NonlinearPDESolver"),
    "IterationState": ("src.problem_solvers.nonlinear_pde_solver", "IterationState"),
    "TimePDESolver": ("src.problem_solvers.nonlinear_pde_solver", "TimePDESolver"),
    "PseudoTimePDESolver": ("src.problem_solvers.nonlinear_pde_solver", "PseudoTimePDESolver"),
    "FixIterationPDESolver": ("src.problem_solvers.nonlinear_pde_solver", "FixIterationPDESolver"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
