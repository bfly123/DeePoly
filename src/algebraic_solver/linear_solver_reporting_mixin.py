import time
from typing import Dict, Union

import numpy as np
import scipy.sparse as sp


class LinearSolverReportingMixin:
    def _normalize_rhs(self, b: np.ndarray) -> np.ndarray:
        return b.reshape(-1, 1) if b.ndim == 1 else b

    def _compute_residual_metrics(
        self,
        A: Union[np.ndarray, sp.spmatrix],
        x: np.ndarray,
        b: np.ndarray,
    ) -> Dict[str, float]:
        residual = A.dot(x) - b if sp.issparse(A) else A @ x - b
        residual_norm = np.linalg.norm(residual)
        b_norm = np.linalg.norm(b)
        return {
            "residual_norm": residual_norm,
            "residual_norm_inf": np.linalg.norm(residual, ord=np.inf),
            "residual_norm_1": np.linalg.norm(residual, ord=1),
            "relative_residual": residual_norm / max(b_norm, 1e-16),
            "b_norm": b_norm,
        }

    def _build_step_stats(
        self,
        A_shape,
        info: Dict,
        method: str,
        solve_time: float,
        residuals: Dict[str, float],
    ) -> Dict:
        return {
            "step": self.step_counter,
            "method": method,
            "matrix_shape": A_shape,
            "condition_number": info.get("condition_number"),
            "residual_norm": residuals["residual_norm"],
            "residual_norm_inf": residuals["residual_norm_inf"],
            "residual_norm_1": residuals["residual_norm_1"],
            "relative_residual": residuals["relative_residual"],
            "b_norm": residuals["b_norm"],
            "solve_time": solve_time,
            "iterations": info.get("iterations"),
            "rank": info.get("rank"),
        }

    def _print_solve_summary(
        self,
        A_shape,
        info: Dict,
        method: str,
        solve_time: float,
        residuals: Dict[str, float],
    ) -> None:
        if not self.verbose:
            return
        print(f"=== Linear Solve Step {self.step_counter} ===")
        print(f"Method: {method}")
        print(f"Matrix size: {A_shape[0]} x {A_shape[1]}")
        print(f"Condition number: {info.get('condition_number', 'N/A')}")
        print("Residual Analysis:")
        print(f"  ||Ax-b||_2 = {residuals['residual_norm']:.6e}")
        print(f"  ||Ax-b||_inf = {residuals['residual_norm_inf']:.6e}")
        print(f"  ||Ax-b||_1 = {residuals['residual_norm_1']:.6e}")
        print(f"  ||b||_2 = {residuals['b_norm']:.6e}")
        print(f"  Relative residual = {residuals['relative_residual']:.6e}")
        print(f"Solve time: {solve_time:.6f}s")
        if "iterations" in info:
            print(f"Iterations: {info['iterations']}")
        if "rank" in info:
            print(f"Matrix rank: {info['rank']}")
        print("-" * 45)

    def _record_performance(
        self,
        method: str,
        A_shape,
        solve_time: float,
        residual_norm: float,
    ) -> None:
        if not self.performance_tracking:
            return
        self.performance_history.append(
            {
                "method": method,
                "matrix_shape": A_shape,
                "solve_time": solve_time,
                "residual_norm": residual_norm,
                "timestamp": time.time(),
            }
        )

    def get_performance_report(self) -> Dict:
        """Get aggregate performance statistics."""
        if not self.performance_tracking or not self.performance_history:
            return {"error": "Performance tracking is disabled or no solves were recorded"}

        methods = {}
        total_time = 0.0
        total_solves = len(self.performance_history)
        for record in self.performance_history:
            method = record["method"]
            stats = methods.setdefault(
                method,
                {
                    "count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "min_time": float("inf"),
                    "max_time": 0.0,
                },
            )
            stats["count"] += 1
            stats["total_time"] += record["solve_time"]
            stats["min_time"] = min(stats["min_time"], record["solve_time"])
            stats["max_time"] = max(stats["max_time"], record["solve_time"])
            total_time += record["solve_time"]

        for stats in methods.values():
            stats["avg_time"] = stats["total_time"] / stats["count"]

        return {
            "total_solves": total_solves,
            "total_time": total_time,
            "avg_time": total_time / total_solves if total_solves > 0 else 0.0,
            "method_stats": methods,
            "backend": "GPU" if self.use_gpu else "CPU",
        }

    def reset_performance_tracking(self):
        """Reset tracked performance history."""
        if self.performance_tracking:
            self.performance_history = []

    def get_step_summary(self, clear_buffer: bool = False) -> Dict:
        """Summarize all solves performed in the current solver step."""
        if not self.current_step_stats:
            return self._build_empty_step_summary()

        summary = {
            "num_solves": len(self.current_step_stats),
            "total_time": sum(stats["solve_time"] for stats in self.current_step_stats),
            "methods_used": list({stats["method"] for stats in self.current_step_stats}),
            "max_residual": max(stats["residual_norm"] for stats in self.current_step_stats),
            "max_relative_residual": max(
                stats["relative_residual"] for stats in self.current_step_stats
            ),
            "max_condition_number": self._get_max_condition_number(),
            "stats": self.current_step_stats,
        }
        if clear_buffer:
            self.current_step_stats = []
        return summary

    def _build_empty_step_summary(self) -> Dict:
        return {
            "num_solves": 0,
            "total_time": 0.0,
            "methods_used": [],
            "max_residual": 0.0,
            "max_relative_residual": 0.0,
        }

    def _get_max_condition_number(self):
        condition_numbers = [
            stats["condition_number"]
            for stats in self.current_step_stats
            if stats["condition_number"] is not None
        ]
        return max(condition_numbers) if condition_numbers else None

    def reset_step_counter(self):
        """Reset step-local solver counters."""
        self.step_counter = 0
        self.current_step_stats = []
        if self.verbose:
            print("=== Linear Solver Step Counter Reset ===")
