import os
from typing import Union

import numpy as np
import scipy.sparse as sp


class LinearSolverRecommendationMixin:
    @staticmethod
    def recommend_method(A: Union[np.ndarray, sp.spmatrix], memory_limit: float = None) -> str:
        """Recommend a solve method from static matrix characteristics."""
        m, n = A.shape
        if LinearSolverRecommendationMixin._is_massive_problem(m, n):
            return "lsmr"
        if sp.issparse(A):
            return LinearSolverRecommendationMixin._recommend_sparse_method(A)
        if LinearSolverRecommendationMixin._exceeds_memory_limit(m, n, memory_limit):
            return "lsmr"
        if LinearSolverRecommendationMixin._prefer_parallel_direct(m, n):
            return "parallel_direct"
        if m > 10 * n:
            return "qr"
        return LinearSolverRecommendationMixin._recommend_small_dense_method(A, m, n)

    @staticmethod
    def _is_massive_problem(m: int, n: int) -> bool:
        return m > 1e6 or n > 1e5

    @staticmethod
    def _recommend_sparse_method(A: sp.spmatrix) -> str:
        m, n = A.shape
        density = A.nnz / (m * n)
        return "lsqr" if density < 0.01 else "lsmr"

    @staticmethod
    def _exceeds_memory_limit(m: int, n: int, memory_limit: float = None) -> bool:
        if memory_limit is None:
            return False
        memory_direct = m * n * 8 / (1024**3)
        return memory_direct > memory_limit

    @staticmethod
    def _prefer_parallel_direct(m: int, n: int) -> bool:
        cpu_cores = os.cpu_count() or 1
        return m > 10 * n and m > 5000 and cpu_cores > 2

    @staticmethod
    def _recommend_small_dense_method(A: np.ndarray, m: int, n: int) -> str:
        if m >= 1e4 or n >= 1e3:
            return "lsmr"
        try:
            return "svd" if np.linalg.cond(A) > 1e10 else "direct"
        except Exception:
            return "direct"
