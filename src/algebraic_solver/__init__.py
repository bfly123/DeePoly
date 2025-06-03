"""
Algebraic solver module, providing various equation solving algorithms
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