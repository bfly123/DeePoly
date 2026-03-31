"""
Computational Fluid Dynamics Hybrid Network Solver Package

This package provides a series of tools and methods for solving fluid dynamics problems.
"""

from importlib import import_module

__all__ = [
    'abstract_class',
    'algebraic_solver',
    'problem_solvers',
    'meta_coding'
]


def __getattr__(name):
    """Lazy-load top-level subpackages to avoid import-time side effects."""
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
