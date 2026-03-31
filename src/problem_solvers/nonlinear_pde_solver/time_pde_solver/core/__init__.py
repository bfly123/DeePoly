"""Lazy exports for time PDE core components."""

from importlib import import_module

_EXPORTS = {
    "TimePDENet": ("src.problem_solvers.nonlinear_pde_solver.time_pde_solver.core.net", "TimePDENet"),
    "TimePDEFitter": ("src.problem_solvers.nonlinear_pde_solver.time_pde_solver.core.fitter", "TimePDEFitter"),
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
