"""
时间相关PDE求解器
"""

# 导入相关类和函数
from .solver import TimePDESolver
from .core import TimePDENet, TimePDEFitter
from .utils import TimePDEConfig, TimePDEDataGenerator, TimePDEVisualizer

__all__ = [
    'TimePDESolver',
    'TimePDENet',
    'TimePDEFitter',
    'TimePDEConfig',
    'TimePDEDataGenerator',
    'TimePDEVisualizer'
] 