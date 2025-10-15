"""
Problem solvers module, including various problem-specific solvers
"""

# Import submodules
from . import func_fitting_solver
from . import linear_pde_solver
from . import nonlinear_pde_solver

# Export solver classes
from .func_fitting_solver.solver import FuncFittingSolver
from .linear_pde_solver import LinearPDESolver
from .nonlinear_pde_solver import (
    NonlinearPDESolver,
    IterationState,
    TimePDESolver,
    PseudoTimePDESolver,
    FixIterationPDESolver
)

__all__ = [
    # Submodules
    'func_fitting_solver',
    'linear_pde_solver',
    'nonlinear_pde_solver',

    # Solver classes
    'FuncFittingSolver',
    'LinearPDESolver',
    'NonlinearPDESolver',
    'IterationState',
    'TimePDESolver',
    'PseudoTimePDESolver',
    'FixIterationPDESolver'
] 