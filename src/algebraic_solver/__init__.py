"""
Algebraic solver module, providing various equation solving algorithms
"""

from importlib import import_module

__all__ = [
    'LinearSolver',
    'FastNewtonSolver',
    'GaussNewtonSolver',
    'TrustRegionSolver'
]

_EXPORT_MAP = {
    "LinearSolver": ".linear_solver",
    "FastNewtonSolver": ".fastnewton",
    "GaussNewtonSolver": ".gauss_newton",
    "TrustRegionSolver": ".trustregionsolver",
}


def __getattr__(name):
    """Lazy-load solver implementations to keep optional backends optional."""
    if name in _EXPORT_MAP:
        module = import_module(_EXPORT_MAP[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
