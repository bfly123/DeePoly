"""
时间相关PDE求解器核心组件
"""

# 导出网络模型
from .net import TimePDENet

# 导出拟合器
from .fitter import TimePDEFitter

__all__ = [
    'TimePDENet',
    'TimePDEFitter'
] 