from typing import Dict, Tuple

import numpy as np

from .linear_solver_backend import CUPY_AVAILABLE, GPU_OUT_OF_MEMORY_ERROR, cp


class LinearSolverGPUMixin:
    def _solve_gpu_batched_lstsq(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Using批Process方式AtGPUUpSolve，防Stop显存溢Exit"""
        _ = kwargs
        if not self.use_gpu:
            raise RuntimeError("请求GPUSolve，但GPUUnavailable")

        m, n = A.shape
        batch_size = self._resolve_gpu_batch_size(n)
        if self.verbose:
            print(f"UsingGPU批Process，Batch size: {batch_size}行")

        num_batches = (m + batch_size - 1) // batch_size
        dtype = np.float32 if self.use_mixed_precision else np.float64
        ATA = np.zeros((n, n), dtype=dtype)
        ATb = np.zeros((n, b.shape[1]), dtype=dtype)

        for batch_index in range(num_batches):
            start_idx = batch_index * batch_size
            end_idx = min((batch_index + 1) * batch_size, m)
            A_batch = A[start_idx:end_idx].astype(dtype)
            b_batch = b[start_idx:end_idx].astype(dtype)
            A_gpu = cp.asarray(A_batch)
            b_gpu = cp.asarray(b_batch)
            ATA += cp.asnumpy(A_gpu.T @ A_gpu)
            ATb += cp.asnumpy(A_gpu.T @ b_gpu)
            del A_gpu, b_gpu
            cp.get_default_memory_pool().free_all_blocks()

        x = np.linalg.solve(ATA, ATb)
        info = {
            "condition_number": np.linalg.cond(ATA),
            "batches": num_batches,
            "batch_size": batch_size,
            "mixed_precision": self.use_mixed_precision,
            "backend": "cupy+numpy",
        }
        return x, info

    def _resolve_gpu_batch_size(self, n_cols: int) -> int:
        if self.gpu_batch_size is not None:
            return self.gpu_batch_size

        try:
            free_memory = cp.cuda.runtime.memGetInfo()[0]
            element_size = 4 if self.use_mixed_precision else 8
            max_rows = int(free_memory * 0.5 / (n_cols * element_size))
            return min(max_rows, 10000)
        except Exception:
            return 5000

    def _solve_gpu_direct(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """UsingCuPy直接SolveLinear最小二乘Problem，Increase了Inner存管理"""
        _ = kwargs
        if not self.use_gpu:
            raise RuntimeError("请求GPUSolve，但GPUUnavailable")

        m, n = A.shape
        use_float32 = self.use_mixed_precision and m * n > 1e6
        dtype = np.float32 if use_float32 else np.float64

        try:
            A_gpu = cp.asarray(A, dtype=dtype)
            b_gpu = cp.asarray(b, dtype=dtype)
            x_gpu = cp.linalg.lstsq(A_gpu, b_gpu, rcond=None)[0]
            try:
                singular_values = cp.linalg.svd(A_gpu, compute_uv=False)
                condition_number = float(singular_values[0] / singular_values[-1])
            except Exception:
                condition_number = None
            x = cp.asnumpy(x_gpu)
            del A_gpu, b_gpu, x_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except GPU_OUT_OF_MEMORY_ERROR:
            if self.verbose:
                print("GPUInner存不足，AttemptUsing批Processmethod")
            return self._solve_gpu_batched_lstsq(A, b)

        info = {
            "condition_number": condition_number,
            "precision": "32bit" if use_float32 else "64bit",
            "backend": "cupy",
        }
        return x, info

    def _solve_gpu_svd(
        self,
        A: np.ndarray,
        b: np.ndarray,
        rcond: float = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """UsingCuPy的SVDDecomposeSolve"""
        _ = kwargs
        if not self.use_gpu:
            raise RuntimeError("请求GPUSolve，但GPUUnavailable")

        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)
        try:
            U, singular_values, Vh = cp.linalg.svd(A_gpu, full_matrices=False)
            rcond = rcond or float(cp.finfo(cp.float64).eps * max(A_gpu.shape))
            s_mask = singular_values > rcond * singular_values[0]
            s_inv = cp.zeros_like(singular_values)
            s_inv[s_mask] = 1 / singular_values[s_mask]
            x_gpu = Vh.T @ (s_inv.reshape(-1, 1) * (U.T @ b_gpu))
            rank = int(cp.sum(s_mask))
            condition_number = (
                float(singular_values[0] / singular_values[s_mask][-1]) if rank > 0 else float("inf")
            )
            x = cp.asnumpy(x_gpu)
            info = {
                "rank": rank,
                "singular_values": cp.asnumpy(singular_values),
                "condition_number": condition_number,
                "backend": "cupy",
            }
            return x, info
        finally:
            del A_gpu, b_gpu
            if "x_gpu" in locals():
                del x_gpu
            if "U" in locals():
                del U, singular_values, Vh, s_inv
            cp.get_default_memory_pool().free_all_blocks()

    def _solve_gpu_qr(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """UsingCuPy的QRDecomposeSolve"""
        _ = kwargs
        if not self.use_gpu:
            raise RuntimeError("请求GPUSolve，但GPUUnavailable")

        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)
        try:
            Q, R = cp.linalg.qr(A_gpu, mode="economic")
            QTb = Q.T @ b_gpu
            x_gpu = cp.linalg.solve_triangular(R, QTb)
            info = {"condition_number": float(cp.linalg.cond(R)), "backend": "cupy"}
            return cp.asnumpy(x_gpu), info
        finally:
            del A_gpu, b_gpu
            if "x_gpu" in locals():
                del x_gpu
            if "Q" in locals():
                del Q, R, QTb
            cp.get_default_memory_pool().free_all_blocks()

    def _solve_gpu_lstsq(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """UsingCuPy的Encapsulation最小二乘functionSolve，针对大型MatrixOptimize"""
        _ = kwargs
        if not self.use_gpu:
            raise RuntimeError("请求GPUSolve，但GPUUnavailable")

        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)
        try:
            x_gpu = cp.linalg.lstsq(A_gpu, b_gpu, rcond=None)[0]
            return cp.asnumpy(x_gpu), {"backend": "cupy"}
        finally:
            del A_gpu, b_gpu
            if "x_gpu" in locals():
                del x_gpu
            cp.get_default_memory_pool().free_all_blocks()

    def _get_available_gpu_memory(self):
        """GetCurrentAvailable的GPUInner存（以GB为单位）"""
        if not self.use_gpu:
            return 0
        try:
            free, _ = cp.cuda.runtime.memGetInfo()
            return free / (1024**3)
        except Exception:
            return None

    def _has_cuda_device(self):
        if not CUPY_AVAILABLE:
            return False
        try:
            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False
