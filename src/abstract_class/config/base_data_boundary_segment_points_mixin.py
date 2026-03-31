from typing import Tuple

import numpy as np


class BaseDataBoundarySegmentPointMixin:
    def _generate_segment_boundary_points(
        self,
        region: str,
        points: int,
        segment_idx: int,
    ) -> np.ndarray:
        if points <= 0:
            return np.array([])

        dim_region = self._parse_dim_region(region)
        if dim_region is not None:
            return self._generate_segment_dim_boundary_points(dim_region, segment_idx, points)

        if self.n_dim == 1:
            return self._generate_segment_1d_named_boundary(region, segment_idx)
        if self.n_dim == 2:
            mapping = {
                "left": (0, False),
                "right": (0, True),
                "bottom": (1, False),
                "top": (1, True),
            }
            if region in mapping:
                return self._generate_segment_dim_boundary_points(mapping[region], segment_idx, points)
        return np.array([])

    def _extract_global_boundary_points_for_segment(
        self,
        region: str,
        points: int,
        segment_idx: int,
    ) -> np.ndarray:
        x_boundary = self._generate_boundary_points(region, points)
        if x_boundary.size == 0:
            return np.array([])
        mask = self._create_segment_mask(x_boundary, segment_idx)
        if not np.any(mask):
            return np.array([])
        return x_boundary[mask]

    def _generate_segment_dim_boundary_points(
        self,
        dim_region: Tuple[int, bool],
        segment_idx: int,
        points: int,
        tol: float = 1e-12,
    ) -> np.ndarray:
        dim_idx, is_max = dim_region
        segment_x_min = np.asarray(self.config.x_min[segment_idx], dtype=np.float64)
        segment_x_max = np.asarray(self.config.x_max[segment_idx], dtype=np.float64)
        domain = np.asarray(self.config.x_domain, dtype=np.float64)

        boundary_value = segment_x_max[dim_idx] if is_max else segment_x_min[dim_idx]
        domain_value = domain[dim_idx][1] if is_max else domain[dim_idx][0]
        if not np.isclose(boundary_value, domain_value, atol=tol):
            return np.array([])

        if self.n_dim == 1:
            return np.array([[boundary_value]], dtype=np.float64)

        if self.n_dim != 2:
            return np.array([])

        other_dim = 1 - dim_idx
        boundary_points = np.zeros((points, self.n_dim), dtype=np.float64)
        boundary_points[:, dim_idx] = boundary_value
        boundary_points[:, other_dim] = np.linspace(
            segment_x_min[other_dim],
            segment_x_max[other_dim],
            points,
        )
        return boundary_points

    def _generate_segment_1d_named_boundary(
        self,
        region: str,
        segment_idx: int,
    ) -> np.ndarray:
        mapping = {"left": (0, False), "right": (0, True)}
        if region not in mapping:
            return np.array([])
        return self._generate_segment_dim_boundary_points(mapping[region], segment_idx, 1)
