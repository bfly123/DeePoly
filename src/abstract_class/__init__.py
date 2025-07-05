"""
Abstract class module, providing base classes and interfaces
"""

# Import configuration related components
from .config.base_config import BaseConfig
from .config.base_data import BaseDataGenerator
from .config.base_visualize import BaseVisualizer

# Do not directly import classes that may cause circular imports in __init__
# from .base_net import BaseNet
# from .base_fitter import BaseDeepPolyFitter

__all__ = [
    'BaseConfig',
    'BaseDataGenerator',
    'BaseVisualizer',
    # Users should import directly from .base_net and .base_fitter
    # 'BaseNet',
    # 'BaseDeepPolyFitter'
] 