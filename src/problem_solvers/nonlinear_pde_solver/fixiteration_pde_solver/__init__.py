"""Fixed-Point Iteration PDE Solver

Solves steady-state PDEs using fixed-point iteration by fixing the nonlinear
term N(U^n) from the previous iteration and solving the resulting linear system.
"""

from .solver import FixIterationPDESolver

__all__ = ["FixIterationPDESolver"]
