import time
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

from .linear_solver_backend import CUPY_AVAILABLE, cp


class LinearSolverMethodSelectionMixin:
    def _check_gpu_capability(self) -> bool:
        """Check whether GPU execution is both available and beneficial."""
        if not CUPY_AVAILABLE:
            return False
        try:
            free_memory_gb = self._get_available_gpu_memory()
            if free_memory_gb is None or free_memory_gb < 1.0:
                self._print_verbose("GPU memory is too small, falling back to CPU")
                return False
            cpu_time, gpu_time = self._benchmark_lstsq_backends()
            use_gpu = gpu_time < cpu_time
            self._print_verbose(
                f"Performance testing: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s"
            )
            self._print_verbose(f"Auto select: {'GPU' if use_gpu else 'CPU'}")
            return use_gpu
        except Exception as exc:
            self._print_verbose(f"GPU capability check failed: {exc}, falling back to CPU")
            return False

    def _benchmark_lstsq_backends(self) -> tuple[float, float]:
        test_size = 1000
        test_a = np.random.rand(test_size, test_size // 2).astype(np.float32)
        test_b = np.random.rand(test_size, 1).astype(np.float32)

        start = time.time()
        np.linalg.lstsq(test_a, test_b, rcond=None)
        cpu_time = time.time() - start

        a_gpu = cp.asarray(test_a)
        b_gpu = cp.asarray(test_b)
        start = time.time()
        cp.linalg.lstsq(a_gpu, b_gpu, rcond=None)
        gpu_time = time.time() - start
        del a_gpu, b_gpu
        cp.get_default_memory_pool().free_all_blocks()
        return cpu_time, gpu_time

    def _resolve_requested_method(self, method: str, A: Union[np.ndarray, sp.spmatrix]) -> str:
        if method and method.startswith("gpu_") and not self.use_gpu:
            return self._demote_gpu_method_to_cpu(method)
        if method not in {"auto", "gpu_auto"}:
            return method
        selected_method = self._auto_select_method(A)
        if method == "gpu_auto":
            return self._promote_cpu_method_to_gpu(selected_method)
        return selected_method

    @staticmethod
    def _demote_gpu_method_to_cpu(method: str) -> str:
        return {
            "gpu_direct": "direct",
            "gpu_svd": "svd",
            "gpu_qr": "qr",
            "gpu_lstsq": "direct",
            "gpu_batched_lstsq": "parallel_direct",
        }.get(method, "direct")

    def _promote_cpu_method_to_gpu(self, method: str) -> str:
        if method.startswith("gpu_") or not self.use_gpu:
            return method
        return {"direct": "gpu_direct", "svd": "gpu_svd", "qr": "gpu_qr"}.get(method, "gpu_direct")

    def _get_fallback_method(
        self,
        method: str,
        A: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    ) -> Optional[str]:
        if method == "direct":
            return None
        if method.startswith("gpu_") and self.use_gpu:
            if A is None:
                return "gpu_batched_lstsq"
            is_sparse = sp.issparse(A)
            if is_sparse:
                return self._select_sparse_cpu_method(A)
            return self._select_cpu_method(A, is_sparse, A.shape[0] * A.shape[1])
        return "direct"

    def _auto_select_method(self, A: Union[np.ndarray, sp.spmatrix]) -> str:
        """Select the solve method based on matrix size, sparsity, and memory."""
        m, n = A.shape
        is_sparse = sp.issparse(A)
        if self._requires_iterative_method(m, n, A, is_sparse):
            return "lsmr"

        problem_size = m * n
        memory_needed_gb = self._estimate_problem_memory_gb(problem_size)
        if self.use_gpu and not is_sparse:
            return self._select_gpu_method(problem_size, memory_needed_gb, A)
        if self._cpu_memory_is_tight(memory_needed_gb, is_sparse):
            self._print_verbose(
                f"Memory need ({memory_needed_gb:.2f}GB) exceeds 80% of available CPU memory"
            )
            return "lsmr"
        return self._select_cpu_method(A, is_sparse, problem_size)

    def _requires_iterative_method(
        self,
        m: int,
        n: int,
        A: Union[np.ndarray, sp.spmatrix],
        is_sparse: bool,
    ) -> bool:
        if m > 1e6 or n > 1e5:
            return True
        return is_sparse and A.nnz / (m * n) < 0.05

    def _estimate_problem_memory_gb(self, problem_size: int) -> float:
        element_size = 4 if self.use_mixed_precision else 8
        return problem_size * element_size * 3 / (1024**3)

    def _cpu_memory_is_tight(self, memory_needed_gb: float, is_sparse: bool) -> bool:
        if is_sparse:
            return False
        return memory_needed_gb > self._get_available_cpu_memory_gb() * 0.8

    def _select_gpu_method(
        self,
        problem_size: int,
        memory_needed_gb: float,
        A: np.ndarray,
    ) -> str:
        gpu_mem_gb = self._get_available_gpu_memory()
        if not gpu_mem_gb:
            return "gpu_direct"
        if memory_needed_gb > gpu_mem_gb * 0.8:
            return self._select_gpu_memory_constrained_method(problem_size, memory_needed_gb, gpu_mem_gb)
        if problem_size > 1e7:
            return "gpu_lstsq"
        if self._is_ill_conditioned(A):
            return "gpu_svd"
        return "gpu_direct"

    def _select_gpu_memory_constrained_method(
        self,
        problem_size: int,
        memory_needed_gb: float,
        gpu_mem_gb: float,
    ) -> str:
        self._print_verbose(
            f"Matrix needs {memory_needed_gb:.2f}GB, GPU memory is {gpu_mem_gb:.2f}GB"
        )
        if memory_needed_gb > gpu_mem_gb * 0.95:
            if problem_size > 5e6 and self.n_jobs > 1:
                return "parallel_direct"
            return "direct"
        return "gpu_batched_lstsq"

    def _get_available_cpu_memory_gb(self) -> float:
        try:
            import psutil

            return psutil.virtual_memory().available / (1024**3)
        except Exception:
            return 8.0

    def _select_cpu_method(
        self,
        A: Union[np.ndarray, sp.spmatrix],
        is_sparse: bool,
        problem_size: int,
    ) -> str:
        if is_sparse:
            return self._select_sparse_cpu_method(A)
        if problem_size > 5e6 and self.n_jobs > 1:
            return self._select_parallel_cpu_method(A, problem_size)
        if problem_size < 1e6:
            return self._select_small_dense_cpu_method(A)
        return "direct"

    def _select_sparse_cpu_method(self, A: sp.spmatrix) -> str:
        density = A.nnz / (A.shape[0] * A.shape[1])
        return "lsqr" if density < 0.01 else "lsmr"

    def _select_parallel_cpu_method(
        self,
        A: Union[np.ndarray, sp.spmatrix],
        problem_size: int,
    ) -> str:
        if not self._is_ill_conditioned(A):
            return "parallel_direct"
        if problem_size < 1e8:
            return "parallel_svd"
        return "parallel_ridge"

    def _select_small_dense_cpu_method(self, A: np.ndarray) -> str:
        if self._is_ill_conditioned(A):
            return "svd"
        if A.shape[0] > 3 * A.shape[1]:
            return "qr"
        return "direct"

    def _is_ill_conditioned(
        self,
        A: np.ndarray,
        sample_size: int = 1000,
        threshold: float = 1e6,
    ) -> bool:
        """Estimate whether the matrix has a large condition number."""
        if sp.issparse(A):
            return False
        m, n = A.shape
        if m < 5000 and n < 1000:
            return self._condition_exceeds_threshold(A, threshold)
        if m > sample_size:
            indices = np.random.choice(m, sample_size, replace=False)
            return self._condition_exceeds_threshold(A[indices], threshold / 10)
        return False

    @staticmethod
    def _condition_exceeds_threshold(matrix: np.ndarray, threshold: float) -> bool:
        try:
            return np.linalg.cond(matrix) > threshold
        except Exception:
            return False

    def _print_verbose(self, message: str) -> None:
        if self.verbose:
            print(message)
