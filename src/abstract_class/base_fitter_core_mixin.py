from typing import Dict, List, Tuple

import numpy as np
from torch import nn

from .operator_factory import create_operator_factory


class BaseFitterCoreMixin:
    def _generate_operator_functions(self):
        """Generate all operator functions once"""
        constants = getattr(self.config, "constants", None)
        if not constants and hasattr(self.config, "const_list"):
            constants = self._flatten_const_list(getattr(self.config, "const_list", []))
        self.operator_factory = create_operator_factory(
            all_derivatives=self.all_derivatives,
            constants=constants,
            optimized=True,
        )
        operator_funcs = self.operator_factory.create_all_operators(self.operator_terms)
        self.L1_func = operator_funcs.get("L1_func", None)
        self.L2_func = operator_funcs.get("L2_func", None)
        self.N_func = operator_funcs.get("N_func", None)
        self.F_func = operator_funcs.get("F_func", None)

    @staticmethod
    def _flatten_const_list(const_list) -> Dict:
        if isinstance(const_list, dict):
            return dict(const_list)
        merged = {}
        if isinstance(const_list, list):
            for item in const_list:
                if isinstance(item, dict):
                    merged.update(item)
        return merged

    def fitter_init(self, model: nn.Module):
        """Initialize and pre-compile operator matrices"""
        self._current_model = model
        self._precompile_all_operators(model)
        return model

    def _precompile_all_operators(self, model: nn.Module):
        """Hierarchical pre-compilation of all operators"""
        n_cols = self.config.n_eqs * self.dgN * self.ns
        for segment_idx in range(self.ns):
            features = self._get_features(segment_idx, model)
            self._features[segment_idx] = features
            self._linear_operators[segment_idx] = {
                "L1": self.L1_func(features),
                "L2": self.L2_func(features),
            }
            self._nonlinear_functions[segment_idx] = {
                "N": self._create_nonlinear_function(features, self.N_func, "N"),
                "F": self._create_nonlinear_function(features, self.F_func, "F"),
            }

        if self.config.problem_type != "func_fitting":
            self._add_constraints(model)

        if self.A and self.b:
            self.A_constraints = np.vstack(self.A)
            self.b_constraints = np.vstack([b_i.reshape(-1, 1) for b_i in self.b])
        else:
            self.A_constraints = np.zeros((0, n_cols))
            self.b_constraints = np.zeros((0, 1))
        self._precompiled = True

    def _create_nonlinear_function(self, features, operator_func, operator_name):
        """Pre-compile nonlinear operators as coefficient-only dependent functions"""
        _ = operator_name

        def coeffs_function(coeffs_nonlinear):
            return operator_func(features, coeffs_nonlinear)

        return coeffs_function

    def has_nonlinear_operators(self):
        """检查非线性算子是否存在"""
        return True

    def fit(self, **kwargs) -> np.ndarray:
        """Solve system using nonlinear solver"""
        return self._solve_system(**kwargs)

    def _solve_system(self, **kwargs) -> np.ndarray:
        """Solve nonlinear system using trust region method"""
        shape = (self.ns, self.config.n_eqs, self.dgN)
        L, r = self._build_jacobian(**kwargs)
        linear_method = getattr(self.config, "linear_method", "auto")
        coeffs = self.solver.solve(
            L,
            r,
            method=linear_method,
            **self._get_linear_solver_kwargs(linear_method),
        )[0]
        return coeffs.reshape(shape)

    def _get_linear_solver_kwargs(self, linear_method: str) -> Dict:
        kwargs: Dict = {}
        method = (linear_method or "").lower()

        if "svd" in method:
            rcond = getattr(self.config, "linear_rcond", None)
            if rcond is not None:
                kwargs["rcond"] = float(rcond)

        if "ridge" in method:
            alpha = getattr(self.config, "linear_alpha", None)
            if alpha is not None:
                kwargs["alpha"] = float(alpha)

        return kwargs

    def _build_jacobian(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Build complete Jacobian matrix"""
        ns = self.ns
        ne = self.config.n_eqs
        dgN = self.dgN
        rows_per_segment = [len(self.data["x_segments_norm"][i]) * ne for i in range(ns)]
        total_rows = sum(rows_per_segment)
        n_cols = ne * dgN * ns
        J1 = np.zeros((total_rows, n_cols), dtype=np.float64)
        b1 = np.zeros((total_rows, 1), dtype=np.float64)

        row_start = 0
        for segment_idx in range(ns):
            row_end = row_start + rows_per_segment[segment_idx]
            col_start = segment_idx * ne * dgN
            col_end = (segment_idx + 1) * ne * dgN
            L, r = self._build_segment_jacobian(segment_idx, **kwargs)
            J1[row_start:row_end, col_start:col_end] = L
            b1[row_start:row_end, 0] = r
            row_start = row_end

        return np.vstack([J1, self.A_constraints]), np.vstack([b1, self.b_constraints])

    def _get_segment_features(
        self,
        x: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
        model: nn.Module,
        derivative: List[int] = None,
    ) -> np.ndarray:
        """Get segment features"""
        n_dims = x.shape[1]
        derivative = derivative or [0] * n_dims
        X_poly = self.feature_generator.sniper_features(x, self.config.poly_degree, derivative)

        scale_factor = 1.0
        for dim, der in enumerate(derivative):
            if der > 0 and (x_max[dim] - x_min[dim]) > 1e-9:
                scale_factor /= (x_max[dim] - x_min[dim]) ** der
            elif der > 0:
                print(f"Warning: Zero range encountered for dimension {dim} during scaling.")
                scale_factor = 0
                break
        X_poly *= scale_factor

        if getattr(self.config, "method", "hybrid") == "hybrid":
            X_nn = self.feature_generator.spotter_features(x, x_min, x_max, model, derivative, self.config.device)
            X_poly = np.hstack([X_poly, X_nn])
        return X_poly

    def _get_features(self, segment_idx: int, model: nn.Module) -> List[np.ndarray]:
        """Get features and derivatives"""
        x = self.data["x_segments_norm"][segment_idx]
        x_min = self.config.x_min[segment_idx]
        x_max = self.config.x_max[segment_idx]
        return [self._get_segment_features(x, x_min, x_max, model, derivative) for derivative in self.derivatives]
