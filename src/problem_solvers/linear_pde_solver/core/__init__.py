"""
LinearPartial differential equationSolve器CoreComponent
"""

# ExportNetworkModel
from .net import LinearPDENet

# ExportFitting器
from .fitter import LinearPDEFitter

__all__ = [
    'LinearPDENet',
    'LinearPDEFitter'
] 