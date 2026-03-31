from typing import Dict, List, Tuple

import numpy as np

from src.abstract_class.shape import concat_segments, safe_segment_slice


class BaseFitterPredictionMixin:
    def construct(
        self,
        data: Dict,
        feature_source,
        coeffs: np.ndarray,
        derivative: List[int] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Make predictions using coefficients"""
        derivative = derivative or [0] * self.config.n_dim
        x_segments = data["x_segments_norm"]
        total_size = sum(seg.shape[0] for seg in x_segments)
        U_pred = np.zeros((total_size, self.config.n_eqs))
        U_segments = []
        start_idx = 0

        for segment_idx in range(self.ns):
            segment_size = x_segments[segment_idx].shape[0]
            end_idx = start_idx + segment_size
            features = self._get_segment_features(
                x_segments[segment_idx],
                self.config.x_min[segment_idx],
                self.config.x_max[segment_idx],
                feature_source,
                derivative,
            )
            segment_pred = np.zeros((segment_size, self.config.n_eqs))
            for eq_idx in range(self.config.n_eqs):
                segment_pred[:, eq_idx] = features @ coeffs[segment_idx, eq_idx, :]
            U_pred[start_idx:end_idx, :] = segment_pred
            U_segments.append(segment_pred)
            start_idx = end_idx
        return U_pred, U_segments

    def global_to_segments(self, U_global: np.ndarray) -> List[np.ndarray]:
        """将全局UArrayConvert为段级List"""
        if U_global is None:
            return [None] * self.ns
        U_segments = []
        start_idx = 0
        for seg_idx in range(self.ns):
            n_points = len(self.data["x_segments_norm"][seg_idx])
            end_idx = start_idx + n_points
            U_segments.append(safe_segment_slice(U_global, start_idx, end_idx, "U_global"))
            start_idx = end_idx
        return U_segments

    def segments_to_global(self, U_segments: List[np.ndarray]) -> np.ndarray:
        """将段级ListConvert为全局UArray"""
        if not U_segments or U_segments[0] is None:
            total_points = sum(len(self.data["x_segments_norm"][i]) for i in range(self.ns))
            return np.zeros((total_points, self.n_eqs))
        return concat_segments(U_segments, "U_segments")

    def has_operator(self, operator_name):
        """算子存在性检查"""
        _ = operator_name
        return True

    def get_operator_info(self, operator_name):
        """Get operator information"""
        func = getattr(self, f"{operator_name}_func", None)
        if func:
            return {
                "exists": True,
                "name": func.operator_name,
                "num_terms": len(func.terms),
                "terms": func.terms,
                "is_nonlinear": getattr(func, "is_nonlinear", False),
                "compiled": hasattr(func, "compiled_terms"),
            }
        return {"exists": False}

    def get_system_info(self):
        """Get system information"""
        info = {
            "linear_operators": [],
            "nonlinear_operators": [],
            "total_equations": self.config.n_eqs,
            "total_segments": self.ns,
            "degrees_of_freedom": self.dgN,
        }
        for op_name in ["L1", "L2", "N", "F"]:
            op_info = self.get_operator_info(op_name)
            if op_info.get("is_nonlinear", False):
                info["nonlinear_operators"].append(op_name)
            else:
                info["linear_operators"].append(op_name)
        return info

    def print_system_summary(self):
        """Print system summary information"""
        info = self.get_system_info()
        print("=== DeePoly System Summary ===")
        print(f"Equations: {info['total_equations']}")
        print(f"Segments: {info['total_segments']}")
        print(f"Degrees of Freedom per segment: {info['degrees_of_freedom']}")
        print(f"Linear Operators: {info['linear_operators']}")
        print(f"Nonlinear Operators: {info['nonlinear_operators']}")
        if info["nonlinear_operators"]:
            print("\nNote: System contains nonlinear operators and will require iterative solving.")
        else:
            print("\nNote: System is linear and can be solved directly.")
        print("=" * 30)

    def _initialize_time_scheme(self, scheme_name: str):
        """Initialize time integration scheme"""
        from src.problem_solvers.nonlinear_pde_solver.time_schemes import create_time_scheme

        self.time_scheme = create_time_scheme(scheme_name, self.config)
        self.time_scheme.set_fitter(self)

    def time_step(
        self,
        U_n: np.ndarray,
        U_seg: List[np.ndarray],
        dt: float,
        coeffs_n: np.ndarray = None,
        **kwargs,
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Execute one time step using the configured time scheme"""
        if self.time_scheme is None:
            raise RuntimeError("Time scheme not initialized. Pass time_scheme to __init__ or call _initialize_time_scheme()")
        if not self._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        self.U_seg_current = U_seg
        U_new, U_seg_new, coeffs = self.time_scheme.time_step(U_n, U_seg, dt, coeffs_n, **kwargs)
        self.U_seg_prev = self.U_seg_current
        self.U_seg_current = U_seg_new
        return U_new, U_seg_new, coeffs

    def _get_source_term(self, segment_idx: int) -> np.ndarray:
        """Get source term for segment"""
        if "source_segments" in self.data:
            return self.data["source_segments"][segment_idx]
        n_points = len(self.data["x_segments"][segment_idx])
        return np.zeros((n_points, self.config.n_eqs))
