"""
计算流体力学混合网络求解器包

该包提供了一系列用于流体力学问题求解的工具和方法。
"""

# 导入各个子模块
from . import abstract_class
from . import algebraic_solver
from . import problem_solvers
from . import meta_coding

__all__ = [
    'abstract_class',
    'algebraic_solver',
    'problem_solvers',
    'meta_coding'
] 