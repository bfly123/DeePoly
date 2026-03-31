from typing import Optional, Tuple

import numpy as np


class BaseDataBoundaryGenerationMixin:
    def _generate_boundary_points(self, region: str, points: int) -> np.ndarray:
        x_domain = self.config.x_domain
        dim_region = self._parse_dim_region(region)
        if dim_region is not None:
            return self._generate_dim_boundary_points(dim_region, x_domain, points)
        if self.n_dim == 1:
            return self._generate_1d_named_boundary(region, x_domain)
        if self.n_dim == 2:
            return self._generate_2d_named_boundary(region, x_domain, points)
        if self.n_dim == 3:
            return self._generate_3d_named_boundary(region, x_domain, points)
        print(f"Warning: Boundary region '{region}' not supported for {self.n_dim}D")
        return np.array([])

    @staticmethod
    def _parse_dim_region(region: str) -> Optional[Tuple[int, bool]]:
        if not region.startswith("dim"):
            return None
        try:
            parts = region.split("_")
            return int(parts[0][3:]), parts[1] == "max"
        except (ValueError, IndexError) as exc:
            print(f"Error parsing boundary region {region}: {exc}")
            return None

    def _generate_dim_boundary_points(
        self,
        dim_region: Tuple[int, bool],
        x_domain,
        points: int,
    ) -> np.ndarray:
        dim_idx, is_max = dim_region
        boundary_value = x_domain[dim_idx][1] if is_max else x_domain[dim_idx][0]
        boundary_points = np.zeros((points, self.n_dim))
        boundary_points[:, dim_idx] = boundary_value
        for dim in range(self.n_dim):
            if dim != dim_idx:
                boundary_points[:, dim] = np.linspace(
                    x_domain[dim][0],
                    x_domain[dim][1],
                    points,
                )
        return boundary_points

    @staticmethod
    def _generate_1d_named_boundary(region: str, x_domain) -> np.ndarray:
        return {
            "left": np.array([[x_domain[0][0]]]),
            "right": np.array([[x_domain[0][1]]]),
        }.get(region, np.array([]))

    @staticmethod
    def _generate_2d_named_boundary(region: str, x_domain, points: int) -> np.ndarray:
        x_min, x_max = x_domain[0][0], x_domain[0][1]
        y_min, y_max = x_domain[1][0], x_domain[1][1]
        if region in {"left", "right"}:
            y_coords = np.linspace(y_min, y_max, points)
            x_val = x_min if region == "left" else x_max
            return np.column_stack((np.full(points, x_val), y_coords))
        if region in {"bottom", "top"}:
            x_coords = np.linspace(x_min, x_max, points)
            y_val = y_min if region == "bottom" else y_max
            return np.column_stack((x_coords, np.full(points, y_val)))
        return np.array([])

    @staticmethod
    def _generate_3d_named_boundary(region: str, x_domain, points: int) -> np.ndarray:
        if region not in {"left", "right"}:
            return np.array([])
        x_min, x_max = x_domain[0][0], x_domain[0][1]
        y_min, y_max = x_domain[1][0], x_domain[1][1]
        z_min, z_max = x_domain[2][0], x_domain[2][1]
        points_per_dim = int(np.sqrt(points))
        x_val = x_min if region == "left" else x_max
        y_coords = np.linspace(y_min, y_max, points_per_dim)
        z_coords = np.linspace(z_min, z_max, points_per_dim)
        y_grid, z_grid = np.meshgrid(y_coords, z_coords)
        return np.column_stack(
            (
                np.full(points_per_dim**2, x_val),
                y_grid.flatten(),
                z_grid.flatten(),
            )
        )

    def _get_boundary_normals(self, region: str, num_points: int) -> np.ndarray:
        normals = np.zeros((num_points, self.n_dim))
        dim_region = self._parse_dim_region(region)
        if dim_region is not None:
            dim_idx, is_max = dim_region
            normals[:, dim_idx] = 1.0 if is_max else -1.0
            return normals
        direction = self._named_boundary_direction(region)
        if direction is not None:
            dim_idx, sign = direction
            normals[:, dim_idx] = sign
        return normals

    def _named_boundary_direction(self, region: str) -> Optional[Tuple[int, float]]:
        mapping = {
            "left": (0, -1.0),
            "right": (0, 1.0),
            "bottom": (1, -1.0),
            "top": (1, 1.0),
            "back": (2, -1.0),
            "front": (2, 1.0),
        }
        direction = mapping.get(region)
        if direction is None or direction[0] >= self.n_dim:
            return None
        return direction
