"""
问题求解器模块，包含各种特定问题的求解器
"""

# 导入子模块
from . import func_fitting_solver
from . import general_pde_solver
from . import linear_pde_solver
from . import time_pde_solver

# 直接导出所有求解器类，简化导入
from .func_fitting_solver.solver import FuncFittingSolver
from .time_pde_solver import TimePDESolver
from .linear_pde_solver import LinearPDESolver

# 注意：以下两个求解器类只声明但尚未实现，仅为保持接口一致
# from .general_pde_solver import GeneralPDESolver

__all__ = [
    # 子模块
    'func_fitting_solver',
    'general_pde_solver',
    'linear_pde_solver',
    'time_pde_solver',
    
    # 求解器类
    'FuncFittingSolver',
    'TimePDESolver',
    'LinearPDESolver'
    # 'GeneralPDESolver'  # 未实现
] 