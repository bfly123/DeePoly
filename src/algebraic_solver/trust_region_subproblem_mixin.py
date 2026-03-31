import numpy as np

from .linear_solver_backend import cp


class TrustRegionSubproblemMixin:
    def _solve_trust_region_subproblem(self, B, g, radius, max_iter=None, tol=1e-8):
        """Solve the CPU trust-region subproblem."""
        n = len(g)
        max_iter = n if max_iter is None else max_iter
        direct_step = self._try_direct_trust_region_step(B, g, radius)
        if direct_step is not None:
            return direct_step
        return self._run_cpu_truncated_cg(B, g, radius, max_iter, tol)

    def _try_direct_trust_region_step(self, B, g, radius):
        eigen_min = self._get_smallest_eigenvalue(B)
        if eigen_min <= 0:
            return None
        try:
            delta = np.linalg.solve(B, -g)
        except np.linalg.LinAlgError:
            return None
        return delta if np.linalg.norm(delta) <= radius else None

    def _get_smallest_eigenvalue(self, matrix):
        from scipy.sparse import linalg as splinalg

        return splinalg.eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0]

    def _run_cpu_truncated_cg(self, B, g, radius, max_iter, tol):
        x = np.zeros_like(g)
        r = g.copy()
        p = -r.copy()
        r_norm_sq = r.dot(r)
        p_norm_sq = p.dot(p)
        x_norm_sq = 0.0

        for _ in range(max_iter):
            Bp = B @ p
            pBp = p.dot(Bp)
            if pBp <= 0:
                return self._solve_boundary_step(x, p, p_norm_sq, x_norm_sq, radius)

            alpha = r_norm_sq / (pBp + 1e-15)
            x_new = x + alpha * p
            if np.linalg.norm(x_new) > radius:
                return self._solve_boundary_step(x, p, p_norm_sq, x_norm_sq, radius)

            x = x_new
            x_norm_sq = x.dot(x)
            r_new = r + alpha * Bp
            r_norm_sq_new = r_new.dot(r_new)
            if np.sqrt(r_norm_sq_new) < tol * np.sqrt(r_norm_sq):
                return x

            beta = r_norm_sq_new / r_norm_sq
            r = r_new
            r_norm_sq = r_norm_sq_new
            p = -r + beta * p
            p_norm_sq = r.dot(r) + beta**2 * p_norm_sq + 2 * beta * r.dot(p)
        return x

    def _solve_trust_region_subproblem_gpu(self, B, g, radius, workspace, max_iter=None, tol=1e-8):
        """Solve the GPU trust-region subproblem."""
        B_gpu, g_gpu = self._prepare_gpu_workspace(B, g, workspace)
        n = len(g)
        max_iter = n if max_iter is None else max_iter

        direct_step = self._try_direct_gpu_trust_region_step(B_gpu, g_gpu, radius)
        if direct_step is not None:
            return direct_step
        return self._run_gpu_truncated_cg(B_gpu, g_gpu, radius, max_iter, tol)

    def _prepare_gpu_workspace(self, B, g, workspace):
        if "B" not in workspace:
            workspace["B"] = cp.asarray(B)
            workspace["g"] = cp.asarray(g)
        else:
            workspace["B"][:] = cp.asarray(B)
            workspace["g"][:] = cp.asarray(g)
        return workspace["B"], workspace["g"]

    def _try_direct_gpu_trust_region_step(self, B_gpu, g_gpu, radius):
        try:
            eigen_min = float(cp.linalg.eigvalsh(B_gpu)[0])
            if eigen_min <= 0:
                return None
            try:
                delta = cp.linalg.solve(B_gpu, -g_gpu)
            except cp.linalg.LinAlgError:
                return None
            if cp.linalg.norm(delta) <= radius:
                return cp.asnumpy(delta)
            return None
        except cp.linalg.LinAlgError:
            return None

    def _run_gpu_truncated_cg(self, B_gpu, g_gpu, radius, max_iter, tol):
        x = cp.zeros_like(g_gpu)
        r = g_gpu.copy()
        p = -r.copy()
        r_norm_sq = float(cp.dot(r, r))
        p_norm_sq = float(cp.dot(p, p))
        x_norm_sq = 0.0

        for _ in range(max_iter):
            Bp = B_gpu @ p
            pBp = float(cp.dot(p, Bp))
            if pBp <= 0:
                return cp.asnumpy(
                    self._solve_boundary_step_gpu(x, p, p_norm_sq, x_norm_sq, radius)
                )

            alpha = r_norm_sq / (pBp + 1e-15)
            x_new = x + alpha * p
            x_new_norm = float(cp.linalg.norm(x_new))
            if x_new_norm > radius:
                return cp.asnumpy(
                    self._solve_boundary_step_gpu(x, p, p_norm_sq, x_norm_sq, radius)
                )

            x = x_new
            x_norm_sq = x_new_norm**2
            r_new = r + alpha * Bp
            r_norm_sq_new = float(cp.dot(r_new, r_new))
            if cp.sqrt(r_norm_sq_new) < tol * cp.sqrt(r_norm_sq):
                return cp.asnumpy(x)

            beta = r_norm_sq_new / r_norm_sq
            r = r_new
            r_norm_sq = r_norm_sq_new
            p = -r + beta * p
            p_norm_sq = float(cp.dot(r, r)) + beta**2 * p_norm_sq + 2 * beta * float(cp.dot(r, p))
        return cp.asnumpy(x)

    def _solve_boundary_step(self, x, p, p_norm_sq, x_norm_sq, radius):
        a = p_norm_sq
        b = 2 * x.dot(p)
        c = x_norm_sq - radius**2
        tau = self._compute_boundary_tau(a, b, c)
        return x + tau * p

    def _solve_boundary_step_gpu(self, x, p, p_norm_sq, x_norm_sq, radius):
        a = p_norm_sq
        b = 2 * float(cp.dot(x, p))
        c = x_norm_sq - radius**2
        tau = self._compute_boundary_tau(a, b, c)
        return x + tau * p

    @staticmethod
    def _compute_boundary_tau(a, b, c):
        disc = max(b**2 - 4 * a * c, 0)
        return (-b + np.sqrt(disc)) / (2 * a)
