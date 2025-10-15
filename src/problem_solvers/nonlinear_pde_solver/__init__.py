"""
NonlinearPDE Solver Package

提供统一的非线性PDE求解器基类和三种具体实现：
- TimePDESolver: 物理时间演化
- PseudoTimePDESolver: 伪时间步迭代至稳态
- FixIterationPDESolver: 不动点迭代

基于统一迭代框架，实现约80%的代码复用率。
"""

from .solver import NonlinearPDESolver, IterationState
from .time_pde_solver import TimePDESolver
from .pseudotime_pde_solver import PseudoTimePDESolver
from .fixiteration_pde_solver import FixIterationPDESolver

__all__ = [
    'NonlinearPDESolver',
    'IterationState',
    'TimePDESolver',
    'PseudoTimePDESolver',
    'FixIterationPDESolver'
]
