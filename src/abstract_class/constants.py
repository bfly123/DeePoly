"""
Global mathematical constants module for DeePoly framework.

This module provides commonly used mathematical constants for use across
the entire DeePoly codebase, ensuring consistency and avoiding NameError
issues when using mathematical constants in various modules.
"""

import math
import numpy as np

# Mathematical constants from math module
pi = math.pi
e = math.e
tau = math.tau if hasattr(math, 'tau') else 2 * math.pi
inf = math.inf
nan = math.nan

# NumPy constants (preferred for numerical computations)
PI = np.pi
E = np.e
EULER_GAMMA = np.euler_gamma
INF = np.inf
NAN = np.nan

# Additional mathematical constants
SQRT_2 = math.sqrt(2)
SQRT_PI = math.sqrt(math.pi)
LOG_2 = math.log(2)
LOG_10 = math.log(10)
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2

# Export commonly used constants
__all__ = [
    'pi', 'e', 'tau', 'inf', 'nan',
    'PI', 'E', 'EULER_GAMMA', 'INF', 'NAN',
    'SQRT_2', 'SQRT_PI', 'LOG_2', 'LOG_10', 'GOLDEN_RATIO'
]