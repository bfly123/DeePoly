import numpy as np
from typing import List, Dict, Tuple, Optional

from .fastnewton_common_mixin import FastNewtonCommonMixin
from .fastnewton_cpu_mixin import FastNewtonCPUMixin
from .fastnewton_gpu_mixin import FastNewtonGPUMixin
from .linear_solver_backend import CUPY_AVAILABLE, cp


class FastNewtonSolver(FastNewtonGPUMixin, FastNewtonCPUMixin, FastNewtonCommonMixin):
    """高效且具有基本Stability保证的非Linear solver"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = bool(use_gpu and CUPY_AVAILABLE)
        self._last_solution = None
        self.min_damping = 1e-12
        self.max_damping = 1e-4

    def solve(
        self,
        A: List,
        b: List,
        equations: Dict,
        variables: Dict,
        build_jacobian_fn,
        shape: Tuple[int, int, int],
        max_iter: int = 200,
        tol: float = 1e-8,
        initial_guess: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """快速但仍具有基本Convergent保证的Solvemethod"""
        if self.use_gpu:
            return self._solve_gpu(
                A,
                b,
                equations,
                variables,
                build_jacobian_fn,
                shape,
                max_iter,
                tol,
                initial_guess,
            )
        else:
            return self._solve_cpu(
                A,
                b,
                equations,
                variables,
                build_jacobian_fn,
                shape,
                max_iter,
                tol,
                initial_guess,
            )
