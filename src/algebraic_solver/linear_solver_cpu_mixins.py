from typing import Callable, Dict, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy.sparse import linalg as splinalg


class LinearSolverCPUMixin:
    def _parallel_chunk_layout(self, row_count: int) -> Tuple[int, int]:
        chunk_size = max(1, min(self.chunk_size, max(1, row_count // self.n_jobs)))
        n_chunks = max(1, min(self.n_jobs, max(1, row_count // chunk_size)))
        return chunk_size, n_chunks

    def _solve_direct(
        self,
        A: Union[np.ndarray, sp.spmatrix],
        b: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """直接最小二乘法，Solve(A^T A)x = A^T b"""
        if sp.issparse(A):
            x, istop, itn, _ = splinalg.lsqr(A, b.flatten(), **kwargs)[:4]
            info = {"iterations": itn, "status": istop}
            return x.reshape(-1, 1), info

        x, _, rank, singular_values = linalg.lstsq(A, b, **kwargs)
        info = {"rank": rank, "singular_values": singular_values}
        return x, info

    def _solve_svd(
        self,
        A: np.ndarray,
        b: np.ndarray,
        rcond: float = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """UsingSVDDecomposeSolve，可以Process病态Problem"""
        _ = kwargs
        U, singular_values, Vh = linalg.svd(A, full_matrices=False)
        rcond = rcond or np.finfo(float).eps * max(A.shape)
        s_mask = singular_values > rcond * singular_values[0]
        s_inv = np.zeros_like(singular_values)
        s_inv[s_mask] = 1 / singular_values[s_mask]
        x = Vh.T @ (s_inv.reshape(-1, 1) * (U.T @ b))

        rank = np.sum(s_mask)
        condition_number = singular_values[0] / singular_values[s_mask][-1] if rank > 0 else np.inf
        info = {
            "rank": rank,
            "singular_values": singular_values,
            "condition_number": condition_number,
        }
        return x, info

    def _solve_qr(self, A: np.ndarray, b: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]:
        """UsingQRDecomposeSolve，比SVD更高效，但对病态ProblemProcess不如SVD"""
        _ = kwargs
        Q, R = linalg.qr(A, mode="economic")
        x = linalg.solve_triangular(R, Q.T @ b)
        info = {"condition_number": np.linalg.cond(R)}
        return x, info

    def _solve_ridge(
        self,
        A: Union[np.ndarray, sp.spmatrix],
        b: np.ndarray,
        alpha: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """岭Regression/TikhonovRegularization，适用于病态Problem"""
        _ = kwargs
        _, n = A.shape

        if sp.issparse(A):
            identity = sp.eye(n)
            A_aug = sp.vstack([A, np.sqrt(alpha) * identity])
            b_aug = np.vstack([b, np.zeros((n, 1))])
            x, istop, itn, _ = splinalg.lsqr(A_aug, b_aug.flatten())[:4]
            info = {"iterations": itn, "status": istop, "alpha": alpha}
            return x.reshape(-1, 1), info

        ATA_reg = A.T @ A + alpha * np.eye(n)
        x = linalg.solve(ATA_reg, A.T @ b, assume_a="pos")
        info = {"alpha": alpha, "condition_number": np.linalg.cond(ATA_reg)}
        return x, info

    def _solve_lsqr(
        self,
        A: Union[np.ndarray, sp.spmatrix],
        b: np.ndarray,
        tol: float = 1e-8,
        iter_lim: int = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """UsingLSQRIterateAlgorithmSolve，适用于大型稀疏System"""
        iter_lim = iter_lim or max(A.shape) * 10
        x, istop, itn, r1norm, r2norm, anorm, acond, _, xnorm, _ = splinalg.lsqr(
            A,
            b.flatten(),
            atol=tol,
            btol=tol,
            iter_lim=iter_lim,
            **kwargs,
        )
        info = {
            "iterations": itn,
            "status": istop,
            "residual_norm_1": r1norm,
            "residual_norm_2": r2norm,
            "matrix_norm": anorm,
            "condition_number_estimate": acond,
            "solution_norm": xnorm,
        }
        return x.reshape(-1, 1), info

    def _solve_lsmr(
        self,
        A: Union[np.ndarray, sp.spmatrix],
        b: np.ndarray,
        tol: float = 1e-8,
        max_iter: int = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """UsingLSMRIterateAlgorithmSolve，Usually比LSQRConvergent更快"""
        max_iter = max_iter or max(A.shape) * 10
        x, istop, itn, normr, _, norma, conda, normx = splinalg.lsmr(
            A,
            b.flatten(),
            atol=tol,
            btol=tol,
            maxiter=max_iter,
            **kwargs,
        )
        info = {
            "iterations": itn,
            "status": istop,
            "residual_norm": normr,
            "matrix_norm": norma,
            "condition_number_estimate": conda,
            "solution_norm": normx,
        }
        return x.reshape(-1, 1), info

    def _process_matrix_chunk(
        self,
        func: Callable,
        A_chunk: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """ProcessMatrix分Block的工作function"""
        return func(A_chunk, *args, **kwargs)

    def _solve_parallel_direct(
        self,
        A: np.ndarray,
        b: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """UsingLine程池并行Solve的直接法"""
        if sp.issparse(A):
            return self._solve_direct(A, b, **kwargs)

        m, _ = A.shape
        if m < self.n_jobs * self.chunk_size:
            return self._solve_direct(A, b, **kwargs)

        _, n_chunks = self._parallel_chunk_layout(m)
        A_chunks = np.array_split(A, n_chunks, axis=0)
        b_chunks = np.array_split(b, n_chunks, axis=0)

        futures_ATb = []
        futures_ATA = []
        for index in range(n_chunks):
            futures_ATb.append(
                self.executor.submit(lambda A_i, b_i: A_i.T @ b_i, A_chunks[index], b_chunks[index])
            )
            futures_ATA.append(self.executor.submit(lambda A_i: A_i.T @ A_i, A_chunks[index]))

        ATb = sum(future.result() for future in futures_ATb)
        ATA = sum(future.result() for future in futures_ATA)
        x = linalg.solve(ATA, ATb, assume_a="pos")
        info = {
            "condition_number": np.linalg.cond(ATA),
            "parallel_chunks": n_chunks,
            "parallel_jobs": self.n_jobs,
        }
        return x, info

    def _solve_parallel_svd(
        self,
        A: np.ndarray,
        b: np.ndarray,
        rcond: float = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """Using并行SVDDecomposeSolve，PassDivideData并行ComputePartialResult"""
        _ = kwargs
        if sp.issparse(A):
            return self._solve_svd(A, b, rcond=rcond)

        m, n = A.shape
        if m < 5000 or n < 1000:
            return self._solve_svd(A, b, rcond=rcond)

        U, singular_values, Vh = linalg.svd(A, full_matrices=False)
        _, n_chunks = self._parallel_chunk_layout(U.shape[0])

        if n_chunks > 1:
            U_chunks = np.array_split(U, n_chunks, axis=0)
            b_chunks = np.array_split(b, n_chunks, axis=0)
            futures = []
            for index in range(n_chunks):
                futures.append(
                    self.executor.submit(lambda U_i, b_i: U_i.T @ b_i, U_chunks[index], b_chunks[index])
                )
            UTb = sum(future.result() for future in futures)
        else:
            UTb = U.T @ b

        rcond = rcond or np.finfo(float).eps * max(A.shape)
        s_mask = singular_values > rcond * singular_values[0]
        s_inv = np.zeros_like(singular_values)
        s_inv[s_mask] = 1 / singular_values[s_mask]
        x = Vh.T @ (s_inv.reshape(-1, 1) * UTb)

        rank = np.sum(s_mask)
        condition_number = singular_values[0] / singular_values[s_mask][-1] if rank > 0 else np.inf
        info = {
            "rank": rank,
            "singular_values": singular_values,
            "condition_number": condition_number,
            "parallel_chunks": n_chunks,
            "parallel_jobs": self.n_jobs,
        }
        return x, info

    def _solve_parallel_ridge(
        self,
        A: np.ndarray,
        b: np.ndarray,
        alpha: float = 1.0,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """并行Implementation的岭Regression/TikhonovRegularization"""
        _ = kwargs
        m, n = A.shape
        _, n_chunks = self._parallel_chunk_layout(m)
        A_chunks = np.array_split(A, n_chunks, axis=0)
        b_chunks = np.array_split(b, n_chunks, axis=0)

        futures_ATb = []
        futures_ATA = []
        for index in range(n_chunks):
            futures_ATb.append(
                self.executor.submit(lambda A_i, b_i: A_i.T @ b_i, A_chunks[index], b_chunks[index])
            )
            futures_ATA.append(self.executor.submit(lambda A_i: A_i.T @ A_i, A_chunks[index]))

        ATb = sum(future.result() for future in futures_ATb)
        ATA = sum(future.result() for future in futures_ATA)
        ATA_reg = ATA + alpha * np.eye(n)
        x = linalg.solve(ATA_reg, ATb, assume_a="pos")
        info = {
            "alpha": alpha,
            "condition_number": np.linalg.cond(ATA_reg),
            "parallel_chunks": n_chunks,
        }
        return x, info
