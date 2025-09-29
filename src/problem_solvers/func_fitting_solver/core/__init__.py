"""
函数拟合求解器核心组件
"""

# 导出网络模型
from .net import FuncFittingNet

# 导出拟合器
from .fitter import FuncFittingFitter

__all__ = [
    'FuncFittingNet',
    'FuncFittingFitter'
] 