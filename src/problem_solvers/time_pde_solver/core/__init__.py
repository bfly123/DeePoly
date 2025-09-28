"""
TimeRelatedPDESolve器CoreComponent
"""

# ExportNetworkModel
from .net import TimePDENet

# ExportFitting器
from .fitter import TimePDEFitter

__all__ = [
    'TimePDENet',
    'TimePDEFitter'
] 