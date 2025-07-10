"""
Meta coding module for automatic code generation
"""

from .auto_snipper import OperatorParser, parse_operators
from .auto_spotter import update_hybrid_fitter_code, update_physics_loss_from_config, update_pytorch_net_code

__all__ = [
    'OperatorParser',
    'parse_operators', 
    'update_hybrid_fitter_code',
    'update_physics_loss_from_config',
    'update_pytorch_net_code'
]