"""Lazy exports for time PDE utility components."""

from importlib import import_module

_EXPORTS = {
    "TimePDEConfig": ("src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils.config", "TimePDEConfig"),
    "TimePDEDataGenerator": ("src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils.data", "TimePDEDataGenerator"),
    "TimePDEVisualizer": ("src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils.visualize", "TimePDEVisualizer"),
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
