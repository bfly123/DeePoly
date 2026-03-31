import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Union

import numpy as np
import scipy.sparse as sp

from .linear_solver_backend import CUPY_AVAILABLE
from .linear_solver_cpu_mixins import LinearSolverCPUMixin
from .linear_solver_gpu_mixins import LinearSolverGPUMixin
from .linear_solver_support_mixins import LinearSolverSupportMixin


class LinearSolver(LinearSolverSupportMixin, LinearSolverGPUMixin, LinearSolverCPUMixin):
    """过定LinearSystemSolve器，用于SolutionAx=bProblem（A的行数Greater than列数），SupportCuPy GPU加速和CPU并行Compute"""

    def __init__(
        self,
        verbose: bool = False,
        use_gpu: bool = True,
        n_jobs: int = None,
        chunk_size: int = 1000,
        gpu_memory_fraction: float = 0.9,
        gpu_batch_size: int = None,
        use_mixed_precision: bool = False,
        performance_tracking: bool = False,
        config: object = None,
    ):
        """Initialize solver"""
        self.verbose = verbose
        self.performance_tracking = performance_tracking
        self.gpu_memory_fraction = gpu_memory_fraction
        self.gpu_batch_size = gpu_batch_size
        self.use_mixed_precision = use_mixed_precision

        use_gpu = self._resolve_gpu_request(use_gpu, config)
        self.n_jobs = n_jobs if n_jobs is not None else max(1, os.cpu_count() - 1)
        self.chunk_size = chunk_size
        self.gpu_available = CUPY_AVAILABLE
        self.use_gpu = self._finalize_gpu_usage(use_gpu)

        if self.verbose:
            backend = "GPU (CuPy)" if self.use_gpu else f"CPU ({self.n_jobs}Core)"
            print(f"ComputeBackwardEnd: {backend}")
            if self.use_mixed_precision:
                print("Using混合PrecisionCompute")

        self.methods = self._build_method_registry()
        self.executor = ThreadPoolExecutor(max_workers=self.n_jobs)
        self._configure_gpu_memory_pool()
        self.performance_history = [] if performance_tracking else None
        self.step_counter = 0
        self.current_step_stats = []

    def solve(
        self,
        A: Union[np.ndarray, sp.spmatrix],
        b: np.ndarray,
        method: str = "gpu_auto",
        fallback: bool = True,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """SolveLinearSystemAx=b"""
        m, n = A.shape
        if m < n:
            raise ValueError(f"InputMatrix不Yes过定System：行数({m}) < 列数({n})")

        b = self._normalize_rhs(b)
        original_method = method
        method = self._resolve_requested_method(method, A)

        if original_method in {"auto", "gpu_auto"} and self.verbose:
            print(f"自动选择method: {method}")
        if method not in self.methods:
            raise ValueError(f"未知的Solvemethod: {method}。Availablemethod: {list(self.methods.keys())}")

        start_time = time.time()
        try:
            x, info = self.methods[method](A, b, **kwargs)
        except Exception as exc:
            fallback_method = self._get_fallback_method(method, A)
            if fallback and fallback_method is not None:
                import warnings

                warnings.warn(f"method {method} Fail: {exc}，AttemptReturn退To{fallback_method}")
                return self.solve(A, b, method=fallback_method, fallback=False, **kwargs)
            raise RuntimeError(f"SolveFail且无法Return退: {exc}") from exc

        solve_time = time.time() - start_time
        info["solve_time"] = solve_time
        info["method"] = method
        info["original_method"] = original_method

        residuals = self._compute_residual_metrics(A, x, b)
        info.update(residuals)
        self.step_counter += 1
        step_stats = self._build_step_stats(A.shape, info, method, solve_time, residuals)
        self.current_step_stats.append(step_stats)
        self._print_solve_summary(A.shape, info, method, solve_time, residuals)
        self._record_performance(method, A.shape, solve_time, residuals["residual_norm"])
        return x, info

    def close(self):
        """Close资源"""
        if hasattr(self, "executor"):
            self.executor.shutdown()
