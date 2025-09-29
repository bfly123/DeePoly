"""
线性偏微分方程求解器核心组件
"""

# 导出网络模型
from .net import LinearPDENet

# 导出拟合器
from .fitter import LinearPDEFitter

__all__ = [
    'LinearPDENet',
    'LinearPDEFitter'
] 