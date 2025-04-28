"""
代数求解器模块，提供各种方程求解算法
"""

from .linear_solver import LinearSolver
from .fastnewton import FastNewtonSolver
from .gauss_newton import GaussNewtonSolver
from .trustregionsolver import TrustRegionSolver

__all__ = [
    'LinearSolver',
    'FastNewtonSolver',
    'GaussNewtonSolver',
    'TrustRegionSolver'
] 