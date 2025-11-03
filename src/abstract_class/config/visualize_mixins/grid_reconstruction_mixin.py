import numpy as np
from scipy.interpolate import griddata
from typing import Tuple, Optional


class GridReconstructionMixin:

    def reconstruct_2d_grid(self, x_scatter: np.ndarray, u_scatter: np.ndarray,
                           n_grid: int = 100, method: str = 'cubic') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_coords = x_scatter[:, 0]
        y_coords = x_scatter[:, 1]

        xi = np.linspace(np.min(x_coords), np.max(x_coords), n_grid)
        yi = np.linspace(np.min(y_coords), np.max(y_coords), n_grid)
        Xi, Yi = np.meshgrid(xi, yi)

        Ui = griddata((x_coords, y_coords), u_scatter.squeeze(), (Xi, Yi), method=method)
        return Xi, Yi, Ui

    def reconstruct_multisegment_grid(self, x_segments: list, u_segments: list, n_grid: int = 100) -> dict:
        reconstructed = {
            'x_grids': [],
            'y_grids': [],
            'u_grids': [],
            'x_bounds': [],
            'y_bounds': []
        }

        for x_seg, u_seg in zip(x_segments, u_segments):
            Xi, Yi, Ui = self.reconstruct_2d_grid(x_seg, u_seg, n_grid)
            reconstructed['x_grids'].append(Xi)
            reconstructed['y_grids'].append(Yi)
            reconstructed['u_grids'].append(Ui)
            reconstructed['x_bounds'].append((np.min(x_seg[:, 0]), np.max(x_seg[:, 0])))
            reconstructed['y_bounds'].append((np.min(x_seg[:, 1]), np.max(x_seg[:, 1])))

        return reconstructed

    def normalize_to_computational(self, x_phys: np.ndarray, bounds: Tuple[float, float],
                                   comp_range: Tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
        x_min, x_max = bounds
        c_min, c_max = comp_range
        return c_min + (x_phys - x_min) * (c_max - c_min) / (x_max - x_min)

    def normalize_to_physical(self, x_comp: np.ndarray, bounds: Tuple[float, float],
                              comp_range: Tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
        x_min, x_max = bounds
        c_min, c_max = comp_range
        return x_min + (x_comp - c_min) * (x_max - x_min) / (c_max - c_min)

    def _map_to_grid(self, x: np.ndarray, y: np.ndarray, u_vals: np.ndarray,
                    x_unique: np.ndarray, y_unique: np.ndarray) -> np.ndarray:
        from scipy.interpolate import griddata

        x_indices = np.searchsorted(x_unique, x)
        y_indices = np.searchsorted(y_unique, y)
        U_grid = np.full((len(y_unique), len(x_unique)), np.nan)
        U_grid[y_indices, x_indices] = u_vals

        if np.any(np.isnan(U_grid)):
            X_mesh, Y_mesh = np.meshgrid(x_unique, y_unique, indexing='xy')
            x_coords = np.column_stack([x, y])
            U_grid = griddata(x_coords, u_vals, (X_mesh, Y_mesh), method='linear')

            if np.any(np.isnan(U_grid)):
                nan_mask = np.isnan(U_grid)
                X_flat = X_mesh.flatten()
                Y_flat = Y_mesh.flatten()
                nan_points = np.column_stack([X_flat[nan_mask.flatten()],
                                             Y_flat[nan_mask.flatten()]])
                U_grid_flat = U_grid.flatten()
                U_grid_flat[nan_mask.flatten()] = griddata(x_coords, u_vals,
                                                          nan_points, method='nearest')
                U_grid = U_grid_flat.reshape(len(y_unique), len(x_unique))

        return U_grid
