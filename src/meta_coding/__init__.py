"""
元编程模块，提供代码自动生成和替换功能
"""

from .auto_eq import parse_equation_to_list, generate_all_derivatives, split_equations
from ..problem_solvers.linear_pde_solver.auto_replace_loss import update_physics_loss_code
from .auto_repalce_nonlinear import update_hybrid_fitter_code

__all__ = [
    'parse_equation_to_list', 
    'generate_all_derivatives',
    'split_equations',
    'update_physics_loss_code',
    'update_hybrid_fitter_code'
]