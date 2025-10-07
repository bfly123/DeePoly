"""
Universal PDE Solver with Pseudo-Time Stepping

Solves arbitrary steady-state PDEs using pseudo-time stepping iteration:
    ∂U/∂τ + R(U) = 0
where R(U) = L1(U) + L2(U)·F(U) + N(U) - S

Features:
- Unified treatment of all derivatives (no distinction between time/space)
- Implicit Euler pseudo-time stepping
- Automatic convergence monitoring
- Adaptive pseudo-time step control
"""

from .solver import GeneralPDESolver

__all__ = ["GeneralPDESolver"]