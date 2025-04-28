"""
时间相关PDE求解器工具组件
"""

# 导出配置类
from .config import TimePDEConfig

# 导出数据生成器
from .data import TimePDEDataGenerator

# 导出可视化工具
from .visualize import TimePDEVisualizer

__all__ = [
    'TimePDEConfig',
    'TimePDEDataGenerator',
    'TimePDEVisualizer'
] 