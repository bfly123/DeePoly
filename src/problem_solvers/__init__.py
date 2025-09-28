"""
ProblemSolve器Module，Include各种特定Problem的Solve器
"""

# Import子Module
from . import func_fitting_solver
from . import general_pde_solver
from . import linear_pde_solver
from . import time_pde_solver

# 直接ExportAllSolve器class，简化Import
from .func_fitting_solver.solver import FuncFittingSolver
from .time_pde_solver import TimePDESolver
from .linear_pde_solver import LinearPDESolver

# Note：以Down两个Solve器class只Declaration但尚未Implementation，仅为MaintainInterface一致
# from .general_pde_solver import GeneralPDESolver

__all__ = [
    # 子Module
    'func_fitting_solver',
    'general_pde_solver',
    'linear_pde_solver',
    'time_pde_solver',
    
    # Solve器class
    'FuncFittingSolver',
    'TimePDESolver',
    'LinearPDESolver'
    # 'GeneralPDESolver'  # 未Implementation
] 