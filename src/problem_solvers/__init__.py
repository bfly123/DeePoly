"""
Problem solvers module, including various problem-specific solvers
"""

# Import submodules
from . import func_fitting_solver
from . import general_pde_solver
from . import linear_pde_solver
from . import time_pde_solver

# Export solver classes
from .func_fitting_solver.solver import FuncFittingSolver
from .time_pde_solver import TimePDESolver
from .linear_pde_solver import LinearPDESolver
from .general_pde_solver import GeneralPDESolver

__all__ = [
    # Submodules
    'func_fitting_solver',
    'general_pde_solver',
    'linear_pde_solver',
    'time_pde_solver',

    # Solver classes
    'FuncFittingSolver',
    'TimePDESolver',
    'LinearPDESolver',
    'GeneralPDESolver'
] 