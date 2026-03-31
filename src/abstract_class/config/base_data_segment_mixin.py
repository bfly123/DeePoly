from typing import Dict, List, Tuple

import numpy as np

from src.abstract_class.shape import ensure_points_eqs


class BaseDataSegmentMixin:
    def _generate_global_points(self, mode: str) -> np.ndarray:
        """Generate global sampling points for train/test modes."""
        if mode == "train":
            if isinstance(self.config.points_domain, (list, tuple)):
                return self._generate_grid_points(self.config.points_domain)
            return self._generate_random_points(self.config.points_domain)
        if isinstance(self.config.points_domain_test, int):
            return self._generate_random_points(
                self.config.points_domain_test,
                margin=0.01,
            )
        return self._generate_grid_points(self.config.points_domain_test, margin=0.01)

    def _generate_random_points(self, num_points: int, margin: float = 0.0) -> np.ndarray:
        points = []
        for dim in range(self.n_dim):
            lower = self.config.x_domain[dim][0] + margin
            upper = self.config.x_domain[dim][1] - margin
            points.append(np.random.uniform(lower, upper, num_points))
        return np.column_stack(points)

    def _generate_grid_points(self, points_per_dim, margin: float = 0.0) -> np.ndarray:
        grids = []
        for dim in range(self.n_dim):
            lower = self.config.x_domain[dim][0] + margin
            upper = self.config.x_domain[dim][1] - margin
            grids.append(np.linspace(lower, upper, points_per_dim[dim]))
        return np.array(np.meshgrid(*grids)).reshape(self.n_dim, -1).T

    @staticmethod
    def _normalize_data(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
        """Normalize data to the [0, 1] range where possible."""
        x = np.asarray(x, dtype=np.float64)
        x_min = np.asarray(x_min, dtype=np.float64)
        x_max = np.asarray(x_max, dtype=np.float64)
        normalized = np.zeros_like(x)
        for dim in range(x.shape[-1]):
            span = x_max[..., dim] - x_min[..., dim]
            if span > 1e-10:
                normalized[..., dim] = (x[..., dim] - x_min[..., dim]) / span
            else:
                normalized[..., dim] = x[..., dim]
        return normalized

    def split_global_points(self, x_global: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split global point clouds into segment-local point clouds."""
        x_segments = []
        masks = []
        for segment_idx in range(self.Ns):
            mask = self._create_segment_mask(x_global, segment_idx)
            x_segments.append(x_global[mask])
            masks.append(mask)
        return x_segments, masks

    def split_global_field(self, masks: List[np.ndarray], u_global: np.ndarray) -> List[np.ndarray]:
        """Split a global field into segment-local fields."""
        u_global = ensure_points_eqs(u_global, "u_global")
        u_segments = []
        for segment_idx in range(self.Ns):
            u_segments.append(u_global[masks[segment_idx]])
        return u_segments

    def _create_segment_mask(self, x: np.ndarray, segment_idx: int) -> np.ndarray:
        """Create a boolean mask selecting points inside a segment."""
        x_min, x_max = self.config.x_min[segment_idx], self.config.x_max[segment_idx]
        tol = 1e-12
        masks = []
        for dim in range(self.n_dim):
            masks.append(x[:, dim] >= x_min[dim] - tol)
            masks.append(x[:, dim] <= x_max[dim] + tol)
        return np.logical_and.reduce(masks)

    def _process_segments(self, x_segments: List[np.ndarray], global_boundary_dict: Dict):
        """Normalize per-segment data and precompute swap/boundary projections."""
        x_segments_norm = []
        x_swap = np.zeros((self.Ns, 2 * self.n_dim, self.Nw, self.n_dim), dtype=np.float64)
        x_swap_norm = np.zeros_like(x_swap)
        boundary_segments_dict = []

        for segment_idx in range(self.Ns):
            x_segments_norm.append(
                self._normalize_data(
                    x_segments[segment_idx],
                    self.config.x_min[segment_idx],
                    self.config.x_max[segment_idx],
                )
            )
            if self.config.points_per_swap > 0:
                x_swap_n = self._generate_swap_points(segment_idx)
                x_swap[segment_idx] = x_swap_n
                x_swap_norm[segment_idx] = self._normalize_swap_data(x_swap_n, segment_idx)
            boundary_segments_dict.append(
                self._process_segment_boundary(global_boundary_dict, segment_idx)
            )

        return x_segments_norm, x_swap, x_swap_norm, boundary_segments_dict

    def _generate_swap_points(self, segment_idx: int) -> np.ndarray:
        """Generate swap points for one segment."""
        x_min, x_max = self.config.x_min, self.config.x_max
        x_swap = np.zeros((2 * self.n_dim, self.Nw, self.n_dim))
        for dim in range(self.n_dim):
            x_swap[2 * dim] = self._generate_swap_boundary_points(x_min[segment_idx], x_max[segment_idx], dim, 0)
            x_swap[2 * dim + 1] = self._generate_swap_boundary_points(x_min[segment_idx], x_max[segment_idx], dim, 1)
        return x_swap

    def _generate_swap_boundary_points(
        self,
        x_min: np.ndarray,
        x_max: np.ndarray,
        dim: int,
        is_upper: int,
    ) -> np.ndarray:
        """Generate swap boundary points for one face of one segment."""
        points = np.zeros((self.Nw, self.n_dim))
        points[:, dim] = x_max[dim] if is_upper else x_min[dim]
        for other_dim in range(self.n_dim):
            if other_dim != dim:
                points[:, other_dim] = np.linspace(x_min[other_dim], x_max[other_dim], self.Nw)
        return points

    def _normalize_swap_data(self, x_swap: np.ndarray, segment_idx: int) -> np.ndarray:
        """Normalize swap points for one segment."""
        x_swap_norm = np.zeros_like(x_swap)
        for face_idx in range(x_swap.shape[0]):
            for point_idx in range(x_swap.shape[1]):
                x_swap_norm[face_idx, point_idx] = self._normalize_data(
                    x_swap[face_idx, point_idx],
                    self.config.x_min[segment_idx],
                    self.config.x_max[segment_idx],
                )
        return x_swap_norm
