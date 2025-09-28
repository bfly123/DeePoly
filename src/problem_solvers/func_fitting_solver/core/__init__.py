"""
functionFittingSolve器CoreComponent
"""

# ExportNetworkModel
from .net import FuncFittingNet

# ExportFitting器
from .fitter import FuncFittingFitter

__all__ = [
    'FuncFittingNet',
    'FuncFittingFitter'
] 