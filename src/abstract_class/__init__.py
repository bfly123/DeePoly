"""
抽象类模块，提供基础类和接口
"""

# 导入配置相关组件
from .config.base_config import BaseConfig
from .config.base_data import BaseDataGenerator
from .config.base_visualize import BaseVisualizer

# 不要在 __init__ 中直接导入可能导致循环的类
# from .base_net import BaseNet
# from .base_fitter import BaseDeepPolyFitter

__all__ = [
    'BaseConfig',
    'BaseDataGenerator',
    'BaseVisualizer',
    # 用户应直接从 .base_net 和 .base_fitter 导入
    # 'BaseNet',
    # 'BaseDeepPolyFitter'
] 