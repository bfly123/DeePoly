"""
线性偏微分方程求解器工具组件
"""

# 导出配置类
from .config import LinearPDEConfig

# 导出数据生成器
from .data import LinearPDEDataGenerator

# 导出可视化工具
from .visualize import LinearPDEVisualizer

__all__ = [
    'LinearPDEConfig',
    'LinearPDEDataGenerator',
    'LinearPDEVisualizer'
] 