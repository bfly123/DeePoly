"""Lazy exports for nonlinear PDE solvers."""

from importlib import import_module

_EXPORTS = {
    "NonlinearPDESolver": ("src.problem_solvers.nonlinear_pde_solver.solver", "NonlinearPDESolver"),
    "IterationState": ("src.problem_solvers.nonlinear_pde_solver.solver", "IterationState"),
    "TimePDESolver": ("src.problem_solvers.nonlinear_pde_solver.time_pde_solver", "TimePDESolver"),
    "PseudoTimePDESolver": ("src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver", "PseudoTimePDESolver"),
    "FixIterationPDESolver": ("src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver", "FixIterationPDESolver"),
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
