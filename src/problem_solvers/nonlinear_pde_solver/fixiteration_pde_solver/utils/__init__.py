"""Lazy exports for fixed-point iteration PDE utilities."""

from importlib import import_module

_EXPORTS = {
    "FixIterationPDEConfig": ("src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.utils.config", "FixIterationPDEConfig"),
    "FixIterationPDEDataGenerator": ("src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.utils.data", "FixIterationPDEDataGenerator"),
    "FixIterationPDEVisualizer": ("src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.utils.visualize", "FixIterationPDEVisualizer"),
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
