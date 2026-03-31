from itertools import product
from typing import List

import numpy as np


class BaseFitterConstraintsMixin:
    def _add_constraints(self, model):
        """Add constraint conditions"""
        self.A, self.b = [], []
        if self.config.problem_type != "func_fitting":
            self._add_boundary_conditions(model)
            self._add_periodic_constraints(model)
            self._add_swap_conditions(model)

    def _add_boundary_conditions(self, model):
        """Add boundary conditions"""
        ne = self.config.n_eqs
        for segment_idx in range(self.ns):
            if "boundary_segments_dict" not in self.data or segment_idx >= len(self.data["boundary_segments_dict"]):
                continue
            segment_boundaries = self.data["boundary_segments_dict"][segment_idx]
            for var_idx in range(len(self.config.vars_list)):
                if var_idx not in segment_boundaries:
                    continue
                self._add_dirichlet_constraints(segment_idx, var_idx, segment_boundaries, model, ne)
                self._add_neumann_constraints(segment_idx, var_idx, segment_boundaries, model, ne)
                self._add_robin_constraints(segment_idx, var_idx, segment_boundaries, model, ne)

    def _add_dirichlet_constraints(self, segment_idx, var_idx, segment_boundaries, model, ne):
        dirichlet_data = segment_boundaries[var_idx].get("dirichlet", {})
        if len(dirichlet_data.get("x", [])) <= 0:
            return
        weight = float(getattr(self.config, "boundary_constraint_weight", 10.0))
        x_bd = dirichlet_data["x"]
        U_bd = dirichlet_data["values"]
        features = self._get_segment_features(
            x_bd,
            self.config.x_min[segment_idx],
            self.config.x_max[segment_idx],
            model,
            [0] * self.config.n_dim,
        )
        constraint = np.zeros((x_bd.shape[0], self.dgN * self.ns * ne))
        start_idx = segment_idx * self.dgN * ne + self.dgN * var_idx
        constraint[:, start_idx : start_idx + self.dgN] = features
        self.A.append(weight * constraint)
        self.b.append(weight * U_bd.flatten())

    def _add_neumann_constraints(self, segment_idx, var_idx, segment_boundaries, model, ne):
        neumann_data = segment_boundaries[var_idx].get("neumann", {})
        if len(neumann_data.get("x", [])) <= 0:
            return
        weight = float(getattr(self.config, "boundary_constraint_weight", 10.0))
        x_bd = neumann_data["x"]
        U_bd = neumann_data["values"]
        normals = neumann_data["normals"]
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
                    self.config.x_min[segment_idx],
                    self.config.x_max[segment_idx],
                    model,
                    derivative,
                )
                constraint = np.zeros((1, self.dgN * self.ns * ne))
                start_idx = segment_idx * self.dgN * ne + self.dgN * var_idx
                constraint[:, start_idx : start_idx + self.dgN] = normal[dim] * features
                self.A.append(weight * constraint)
                self.b.append(weight * np.array([U_bd[pt_idx, 0]]))

    def _add_robin_constraints(self, segment_idx, var_idx, segment_boundaries, model, ne):
        robin_data = segment_boundaries[var_idx].get("robin", {})
        if len(robin_data.get("x", [])) <= 0:
            return
        weight = float(getattr(self.config, "boundary_constraint_weight", 10.0))
        x_bd = robin_data["x"]
        U_bd = robin_data["values"]
        normals = robin_data["normals"]
        params = robin_data["params"]
        if not isinstance(params, list) or len(params) < 2:
            params = [1.0, 0.0]
        alpha, beta = params

        for pt_idx in range(x_bd.shape[0]):
            point = x_bd[pt_idx : pt_idx + 1]
            normal = normals[pt_idx]
            features_U = self._get_segment_features(
                point,
                self.config.x_min[segment_idx],
                self.config.x_max[segment_idx],
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
                    self.config.x_min[segment_idx],
                    self.config.x_max[segment_idx],
                    model,
                    derivative,
                )
                features_dU_dn += normal[dim] * features_dim

            constraint = np.zeros((1, self.dgN * self.ns * ne))
            start_idx = segment_idx * self.dgN * ne + self.dgN * var_idx
            constraint[:, start_idx : start_idx + self.dgN] = alpha * features_U + beta * features_dU_dn
            self.A.append(weight * constraint)
            self.b.append(weight * np.array([U_bd[pt_idx, 0]]))

    def _add_periodic_constraints(self, model):
        """Add periodic boundary conditions"""
        if "global_boundary_dict" not in self.data:
            return
        global_boundary_data = self.data["global_boundary_dict"]
        for var_idx in range(len(self.config.vars_list)):
            if var_idx not in global_boundary_data:
                continue
            if "periodic" not in global_boundary_data[var_idx]:
                continue
            for pair in global_boundary_data[var_idx]["periodic"]["pairs"]:
                segment_1 = pair["segment_1"]
                segment_2 = pair["segment_2"]
                x_boundary_1_local = self._normalize_coords(pair["x_1"], segment_1)
                x_boundary_2_local = self._normalize_coords(pair["x_2"], segment_2)
                self._add_periodic_constraints_for_pair(model, segment_1, segment_2, x_boundary_1_local, x_boundary_2_local)

    def _normalize_coords(self, x_global: np.ndarray, segment_idx: int) -> np.ndarray:
        """将全局coordinateConvert为segment的局部Normalizedcoordinate"""
        x_min = np.asarray(self.config.x_min[segment_idx], dtype=np.float64)
        x_max = np.asarray(self.config.x_max[segment_idx], dtype=np.float64)
        x = np.asarray(x_global, dtype=np.float64)
        normalized = np.zeros_like(x)
        for i in range(x.shape[-1]):
            if x_max[..., i] - x_min[..., i] > 1e-10:
                normalized[..., i] = (x[..., i] - x_min[..., i]) / (x_max[..., i] - x_min[..., i])
            else:
                normalized[..., i] = x[..., i]
        return normalized

    def _add_periodic_constraints_for_pair(self, model, segment_1: int, segment_2: int, x_boundary_1_local: np.ndarray, x_boundary_2_local: np.ndarray):
        """Add periodic boundary constraints for a pair using unified constraint builder"""
        weight = float(getattr(self.config, "periodic_constraint_weight", 10.0))
        for eq_idx in range(self.config.n_eqs):
            for derivative in self._iter_equation_derivatives(eq_idx):
                self._add_dual_boundary_constraint(
                    model=model,
                    seg1=segment_1,
                    x1_local=x_boundary_1_local,
                    seg2=segment_2,
                    x2_local=x_boundary_2_local,
                    eq_idx=eq_idx,
                    derivative=derivative,
                    weight=weight,
                )

    def _add_dual_boundary_constraint(
        self,
        model,
        seg1: int,
        x1_local: np.ndarray,
        seg2: int,
        x2_local: np.ndarray,
        eq_idx: int,
        derivative: List[int],
        weight: float = 10.0,
    ):
        """Unified function for adding dual boundary equality constraints"""
        n1, n2 = x1_local.shape[0], x2_local.shape[0]
        assert n1 == n2, f"Boundary points must match: seg{seg1} has {n1}, seg{seg2} has {n2}"
        P1 = self._get_segment_features(x1_local, self.config.x_min[seg1], self.config.x_max[seg1], model, derivative)
        P2 = self._get_segment_features(x2_local, self.config.x_min[seg2], self.config.x_max[seg2], model, derivative)
        cont = np.zeros((n1, self.dgN * self.ns * self.config.n_eqs), dtype=np.float64)
        ndisp = seg1 * self.dgN * self.config.n_eqs + self.dgN * eq_idx
        cont[:, ndisp : ndisp + self.dgN] = P1[:n1, :]
        ndisp = seg2 * self.dgN * self.config.n_eqs + self.dgN * eq_idx
        cont[:, ndisp : ndisp + self.dgN] = -P2[:n1, :]
        self.A.append(weight * cont)
        self.b.append(weight * np.zeros((n1, 1)))

    def _iter_equation_derivatives(self, eq_idx: int):
        """Generate derivative combinations for given equation"""
        max_orders = self.max_derivatives[eq_idx]
        reduced_orders = [max(0, order - 1) for order in max_orders]
        ranges = [range(reduced_order + 1) for reduced_order in reduced_orders]
        for derivative_tuple in product(*ranges):
            yield list(derivative_tuple)

    def _add_swap_conditions(self, model):
        """Add interface continuity conditions"""
        for dim in range(len(self.n_segments)):
            current_segments = list(self.n_segments)
            current_segments[dim] -= 1
            indices = np.array(np.meshgrid(*[range(n) for n in current_segments])).T.reshape(-1, len(self.n_segments))
            for idx in indices:
                self._add_dimension_swap(idx, dim, model)

    def _add_dimension_swap(self, idx: np.ndarray, dim: int, model):
        """Add interface continuity conditions for specified dimension"""
        idx1 = idx.copy()
        idx2 = idx.copy()
        idx2[dim] += 1
        seg1 = np.ravel_multi_index(idx1, self.n_segments, order="F")
        seg2 = np.ravel_multi_index(idx2, self.n_segments, order="F")
        x1_local = self.data["x_swap_norm"][seg1, 2 * dim + 1, :, :]
        x2_local = self.data["x_swap_norm"][seg2, 2 * dim, :, :]
        weight = float(getattr(self.config, "swap_constraint_weight", 10.0))
        for eq_idx in range(self.config.n_eqs):
            for derivative in self._iter_equation_derivatives(eq_idx):
                self._add_dual_boundary_constraint(
                    model=model,
                    seg1=seg1,
                    x1_local=x1_local,
                    seg2=seg2,
                    x2_local=x2_local,
                    eq_idx=eq_idx,
                    derivative=derivative,
                    weight=weight,
                )
