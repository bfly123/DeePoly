from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch import nn
from itertools import product

from .operator_factory import create_operator_factory


class BaseDeepPolyFitter(ABC):
    """Abstract base class for DeePoly fitters"""

    def __init__(self, config, dt: float = None, data: Dict = None):
        from src.abstract_class.features_generator import FeatureGenerator

        self.config = config
        self.data = data
        self.dt = dt

        # Base dimensions
        self.n_dim = config.n_dim
        if hasattr(config, "n_segments"):
            if isinstance(config.n_segments, (list, tuple)):
                self.n_segments = config.n_segments
            else:
                self.n_segments = [config.n_segments] * self.n_dim
        else:
            self.n_segments = [10] * self.n_dim

        self.ns = np.prod(self.n_segments)

        # Pre-compilation cache system
        self._features = {}  # Cache features: {segment_idx: features_list}
        self._linear_operators = {}  # Cache linear operators: {segment_idx: {'L1': matrix, 'L2': matrix}}
        self._nonlinear_functions = {}  # Pre-compiled coefficient functions: {segment_idx: {'N': func, 'F': func}}
        self._precompiled = False  # Pre-compilation completion flag

        self.feature_generator = FeatureGenerator(self.config.linear_device)
        
        # 优先使用config直接属性，回退到operator_parse字典访问
        if hasattr(config, 'max_derivative_orders'):
            self.max_derivatives = config.max_derivative_orders
        else:
            self.max_derivatives = config.operator_parse["max_derivative_orders"]
            
        if hasattr(config, 'all_derivatives'):
            self.all_derivatives = config.all_derivatives
        else:
            self.all_derivatives = config.operator_parse["all_derivatives"]
            
        if hasattr(config, 'derivatives'):
            self.derivatives = config.derivatives
        else:
            self.derivatives = config.operator_parse["derivatives"]
            
        if hasattr(config, 'operator_terms'):
            self.operator_terms = config.operator_terms
        else:
            self.operator_terms = config.operator_parse["operator_terms"]
            
        self.L1 = self.operator_terms.get("L1", None)
        self.L2 = self.operator_terms.get("L2", None)
        self.N = self.operator_terms.get("N", None)
        self.F = self.operator_terms.get("F", None)
        
        self.dg = np.int32(
            np.prod(
                self.config.poly_degree + np.ones(np.shape(self.config.poly_degree))
            )
        )
        if self.config.method == "hybrid":
            self.dgN = self.config.DNN_degree + self.dg
        else:
            self.dgN = self.dg
        self.device = self.config.linear_device
        self.n_eqs = self.config.n_eqs

        # Initialize constraint members
        self.A_constraints = None
        self.b_constraints = None
        self.A = None
        self.b = None

        # Generate operator functions
        self._generate_operator_functions()

    def _generate_operator_functions(self):
        """Generate all operator functions once"""
        constants = getattr(self.config, "constants", {})

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

    @abstractmethod
    def _build_segment_jacobian(
        self,
        segment_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build Jacobian matrix for single segment"""
        pass

    def fitter_init(self, model: nn.Module):
        """Initialize and pre-compile operator matrices"""
        # 保持模型在原设备上，不要移动到linear_device
        self._current_model = model
        self._precompile_all_operators(model)
        return model

    def _precompile_all_operators(self, model: nn.Module):
        """Hierarchical pre-compilation of all operators"""
        n_cols = self.config.n_eqs * self.dgN * self.ns

        for segment_idx in range(self.ns):
            # Level 1: Pre-compile and cache features
            features = self._get_features(segment_idx, model)
            self._features[segment_idx] = features

            # Level 2: Pre-compile linear operators
            linear_ops = {}
            if self.has_operator("L1"):
                linear_ops["L1"] = self.L1_func(features)
            if self.has_operator("L2"):
                linear_ops["L2"] = self.L2_func(features)
            self._linear_operators[segment_idx] = linear_ops

            # Level 3: Pre-compile nonlinear operators
            nonlinear_funcs = {}
            if self.has_operator("N"):
                nonlinear_funcs["N"] = self._create_nonlinear_function(
                    features, self.N_func, "N"
                )
            if self.has_operator("F"):
                nonlinear_funcs["F"] = self._create_nonlinear_function(
                    features, self.F_func, "F"
                )
            self._nonlinear_functions[segment_idx] = nonlinear_funcs

        # Level 4: Pre-compile constraint conditions
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
        def coeffs_function(coeffs_nonlinear):
            """Pre-compiled coefficient function"""
            result = operator_func(features, coeffs_nonlinear)
            return result
        return coeffs_function

    def has_nonlinear_operators(self):
        """Check if nonlinear operators exist"""
        return self.has_operator("N") or self.has_operator("F")

    def fit(self, **kwargs) -> np.ndarray:
        """Solve system using nonlinear solver"""
        coeffs = self._solve_system(**kwargs)
        return coeffs

    def _solve_system(self, **kwargs) -> np.ndarray:
        """Solve nonlinear system using trust region method"""
        shape = (self.ns, self.config.n_eqs, self.dgN)
        L, r = self._build_jacobian(**kwargs)
        coeffs = self.solver.solve(L, r, method="auto")[0]
        return coeffs.reshape(shape)

    def _build_jacobian(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Build complete Jacobian matrix"""
        ns = self.ns
        ne = self.config.n_eqs
        dgN = self.dgN

        # Calculate total number of rows
        total_rows = 0
        rows_per_segment = []
        for i in range(ns):
            n_points = len(self.data["x_segments_norm"][i])
            segment_rows = n_points * ne
            rows_per_segment.append(segment_rows)
            total_rows += segment_rows

        n_cols = ne * dgN * ns

        # Pre-allocate matrices
        J1 = np.zeros((total_rows, n_cols), dtype=np.float64)
        b1 = np.zeros((total_rows, 1), dtype=np.float64)

        # Process each segment
        row_start = 0
        for i in range(ns):
            row_end = row_start + rows_per_segment[i]
            col_start = i * ne * dgN
            col_end = (i + 1) * ne * dgN

            L, r = self._build_segment_jacobian(i, **kwargs)

            # Fill matrices
            J1[row_start:row_end, col_start:col_end] = L
            b1[row_start:row_end, 0] = r

            row_start = row_end

        # Assemble final results
        J = np.vstack([J1, self.A_constraints])
        r = np.vstack([b1, self.b_constraints])

        return J, r

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
        if derivative is None:
            derivative = [0] * n_dims

        X_poly = self.feature_generator.sniper_features(
            x, self.config.poly_degree, derivative
        )

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
            X_nn = self.feature_generator.spotter_features(
                x, x_min, x_max, model, derivative, self.config.device
            )
            X_poly = np.hstack([X_poly, X_nn])

        return X_poly

    def _get_features(self, segment_idx: int, model: nn.Module) -> List[np.ndarray]:
        """Get features and derivatives"""
        feature_list = self.derivatives
        x = self.data["x_segments_norm"][segment_idx]
        x_min = self.config.x_min[segment_idx]
        x_max = self.config.x_max[segment_idx]

        features = []
        for derivative in feature_list:
            features.append(
                self._get_segment_features(x, x_min, x_max, model, derivative)
            )
        return features

    def _add_constraints(self, model: nn.Module):
        """Add constraint conditions"""
        self.A, self.b = [], []
        if self.config.problem_type != "func_fitting":
            self._add_boundary_conditions(model)
            self._add_swap_conditions(model)

    def _add_boundary_conditions(self, model: nn.Module):
        """Add boundary conditions"""
        ne = self.config.n_eqs

        for i in range(self.ns):
            if "boundary_segments_dict" not in self.data or i >= len(
                self.data["boundary_segments_dict"]
            ):
                continue

            segment_boundaries = self.data["boundary_segments_dict"][i]

            for var_idx in range(len(self.config.vars_list)):
                if var_idx not in segment_boundaries:
                    continue

                # Process Dirichlet boundary conditions
                if (
                    "dirichlet" in segment_boundaries[var_idx]
                    and len(segment_boundaries[var_idx]["dirichlet"]["x"]) > 0
                ):
                    x_bd = segment_boundaries[var_idx]["dirichlet"]["x"]
                    U_bd = segment_boundaries[var_idx]["dirichlet"]["values"]

                    features = self._get_segment_features(
                        x_bd,
                        self.config.x_min[i],
                        self.config.x_max[i],
                        model,
                        [0] * self.config.n_dim,
                    )

                    constraint = np.zeros((x_bd.shape[0], self.dgN * self.ns * ne))
                    start_idx = i * self.dgN * ne + self.dgN * var_idx
                    constraint[:, start_idx : start_idx + self.dgN] = features

                    self.A.append(10 * constraint)
                    self.b.append(10 * U_bd.flatten())

                # Process Neumann boundary conditions
                if (
                    "neumann" in segment_boundaries[var_idx]
                    and len(segment_boundaries[var_idx]["neumann"]["x"]) > 0
                ):
                    x_bd = segment_boundaries[var_idx]["neumann"]["x"]
                    U_bd = segment_boundaries[var_idx]["neumann"]["values"]
                    normals = segment_boundaries[var_idx]["neumann"]["normals"]

                    for pt_idx in range(x_bd.shape[0]):
                        point = x_bd[pt_idx : pt_idx + 1]
                        normal = normals[pt_idx]

                        for dim in range(self.config.n_dim):
                            if abs(normal[dim]) < 1e-10:
                                continue

                            derivative = [0] * self.config.n_dim
                            derivative[dim] = 1

                            features = self._get_segment_features(
                                point,
                                self.config.x_min[i],
                                self.config.x_max[i],
                                model,
                                derivative,
                            )

                            constraint = np.zeros((1, self.dgN * self.ns * ne))
                            start_idx = i * self.dgN * ne + self.dgN * var_idx
                            constraint[:, start_idx : start_idx + self.dgN] = (
                                normal[dim] * features
                            )

                            self.A.append(10 * constraint)
                            self.b.append(10 * np.array([U_bd[pt_idx, 0]]))

                # Process Robin boundary conditions
                if (
                    "robin" in segment_boundaries[var_idx]
                    and len(segment_boundaries[var_idx]["robin"]["x"]) > 0
                ):
                    x_bd = segment_boundaries[var_idx]["robin"]["x"]
                    U_bd = segment_boundaries[var_idx]["robin"]["values"]
                    normals = segment_boundaries[var_idx]["robin"]["normals"]
                    params = segment_boundaries[var_idx]["robin"]["params"]

                    if not isinstance(params, list) or len(params) < 2:
                        params = [1.0, 0.0]

                    alpha, beta = params

                    for pt_idx in range(x_bd.shape[0]):
                        point = x_bd[pt_idx : pt_idx + 1]
                        normal = normals[pt_idx]

                        features_U = self._get_segment_features(
                            point,
                            self.config.x_min[i],
                            self.config.x_max[i],
                            model,
                            [0] * self.config.n_dim,
                        )

                        features_dU_dn = np.zeros_like(features_U)

                        for dim in range(self.config.n_dim):
                            if abs(normal[dim]) < 1e-10:
                                continue

                            derivative = [0] * self.config.n_dim
                            derivative[dim] = 1

                            features_dim = self._get_segment_features(
                                point,
                                self.config.x_min[i],
                                self.config.x_max[i],
                                model,
                                derivative,
                            )

                            features_dU_dn += normal[dim] * features_dim

                        constraint = np.zeros((1, self.dgN * self.ns * ne))
                        start_idx = i * self.dgN * ne + self.dgN * var_idx
                        constraint[:, start_idx : start_idx + self.dgN] = (
                            alpha * features_U + beta * features_dU_dn
                        )

                        self.A.append(10 * constraint)
                        self.b.append(10 * np.array([U_bd[pt_idx, 0]]))

    def _add_swap_conditions(self, model: nn.Module):
        """Add interface continuity conditions"""
        n_dims = len(self.n_segments)

        for dim in range(n_dims):
            current_segments = list(self.n_segments)
            current_segments[dim] -= 1

            indices = np.array(
                np.meshgrid(*[range(n) for n in current_segments])
            ).T.reshape(-1, n_dims)

            for idx in indices:
                self._add_dimension_swap(idx, dim, model)

    def _add_dimension_swap(self, idx: np.ndarray, dim: int, model: nn.Module):
        """Add interface continuity conditions for specified dimension"""
        n_dims = len(self.n_segments)
        NS = np.prod(self.n_segments)
        nw = self.config.points_per_swap
        dgN = self.dgN
        ne = self.config.n_eqs

        idx1 = idx.copy()
        idx2 = idx.copy()
        idx2[dim] += 1

        seg1 = np.ravel_multi_index(idx1, self.n_segments, order="F")
        seg2 = np.ravel_multi_index(idx2, self.n_segments, order="F")

        for i in range(ne):
            max_orders = self.max_derivatives[i]
            reduced_orders = [max(0, order - 1) for order in max_orders]
            ranges = [range(reduced_order + 1) for reduced_order in reduced_orders]

            for derivative_tuple in product(*ranges):
                derivative = list(derivative_tuple)
                self._add_swap_derivative(
                    model, seg1, seg2, NS, nw, dgN, i, derivative, dim
                )

    def _add_swap_derivative(
        self,
        model: nn.Module,
        seg1: int,
        seg2: int,
        NS: int,
        nw: int,
        dgN: int,
        eq_idx: int,
        derivative: List[int],
        dim: int,
    ):
        """Add derivative interface conditions for specified dimension"""
        P1 = self._get_segment_features(
            self.data["x_swap_norm"][seg1, 2 * dim + 1, :, :],
            self.config.x_min[seg1],
            self.config.x_max[seg1],
            model,
            derivative,
        )

        P2 = self._get_segment_features(
            self.data["x_swap_norm"][seg2, 2 * dim, :, :],
            self.config.x_min[seg2],
            self.config.x_max[seg2],
            model,
            derivative,
        )

        cont = np.zeros((nw, dgN * NS * self.config.n_eqs), dtype=np.float64)

        ndisp = seg1 * dgN * self.config.n_eqs + dgN * eq_idx
        cont[:, ndisp : ndisp + dgN] = P1

        ndisp = seg2 * dgN * self.config.n_eqs + dgN * eq_idx
        cont[:, ndisp : ndisp + dgN] = -P2

        self.A.extend(10 * cont)
        self.b.extend(10 * np.zeros((nw, 1)).flatten())

    def construct(
        self,
        data: Dict,
        model: nn.Module,
        coeffs: np.ndarray,
        derivative: List[int] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Make predictions using coefficients"""
        x_segments = data["x_segments_norm"]
        x_min = self.config.x_min
        x_max = self.config.x_max
        ne = self.config.n_eqs
        ns = self.ns

        if derivative is None:
            derivative = [0] * self.config.n_dim

        total_size = sum(seg.shape[0] for seg in x_segments)
        U_pred = np.zeros((total_size, ne))
        U_segments = []

        start_idx = 0
        for i in range(ns):
            segment_size = x_segments[i].shape[0]
            end_idx = start_idx + segment_size

            features = self._get_segment_features(
                x_segments[i], x_min[i], x_max[i], model, derivative
            )

            segment_pred = np.zeros((segment_size, ne))
            for j in range(ne):
                segment_pred[:, j] = features @ coeffs[i, j, :]

            U_pred[start_idx:end_idx, :] = segment_pred
            U_segments.append(segment_pred)
            start_idx = end_idx

        return U_pred, U_segments

    def global_to_segments(self, U_global: np.ndarray) -> List[np.ndarray]:
        """将全局U数组转换为段级列表
        
        Args:
            U_global: 全局解数组，形状为 (total_points,) 或 (total_points, n_eqs)
            
        Returns:
            List[np.ndarray]: 段级解值列表，每个元素对应一个段的解值
        """
        if U_global is None:
            return [None] * self.ns
            
        U_segments = []
        start_idx = 0
        for seg_idx in range(self.ns):
            n_points = len(self.data["x_segments_norm"][seg_idx])
            end_idx = start_idx + n_points
            
            if U_global.ndim == 1:
                U_segments.append(U_global[start_idx:end_idx].copy())
            else:
                U_segments.append(U_global[start_idx:end_idx, :].copy())
                
            start_idx = end_idx
            
        return U_segments

    def segments_to_global(self, U_segments: List[np.ndarray]) -> np.ndarray:
        """将段级列表转换为全局U数组
        
        Args:
            U_segments: 段级解值列表
            
        Returns:
            np.ndarray: 全局解数组
        """
        if not U_segments or U_segments[0] is None:
            total_points = sum(len(self.data["x_segments_norm"][i]) for i in range(self.ns))
            return np.zeros(total_points)
            
        return np.concatenate(U_segments)

    def has_operator(self, operator_name):
        """Check if operator exists"""
        return getattr(self, f"{operator_name}_func", None) is not None

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
            if self.has_operator(op_name):
                op_info = self.get_operator_info(op_name)
                if op_info["is_nonlinear"]:
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
