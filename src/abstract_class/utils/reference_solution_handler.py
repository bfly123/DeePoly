import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.interpolate import griddata, interp1d


@dataclass
class ComparisonPayload:
    """Normalized data structure for visualization comparison

    Attributes:
        numerical_coords: Numerical solution coordinates (n_points, n_dim)
        numerical_values: Numerical solution values (n_points,)
        reference_coords: Reference solution coordinates (n_ref_points, n_dim)
        reference_values: Reference solution values (n_ref_points,)
        axis_labels: Labels for each dimension ['x', 't']
        interpolation_method: Method used for interpolation
        time_slice_info: Info about time slicing if applied
    """
    numerical_coords: np.ndarray
    numerical_values: np.ndarray
    reference_coords: np.ndarray
    reference_values: np.ndarray
    axis_labels: list
    interpolation_method: str = 'linear'
    time_slice_info: Optional[Dict] = None


class ReferenceSolutionHandler:
    """Handle dimension matching and interpolation between numerical and reference solutions"""

    @staticmethod
    def detect_dimensions(ref_solution: Dict, numerical_coords: np.ndarray) -> Tuple[int, int, list]:
        """Detect dimensionality of reference and numerical solutions

        Args:
            ref_solution: Reference solution dictionary
            numerical_coords: Numerical solution coordinates (n_points, n_dim)

        Returns:
            (n_dim_numerical, n_dim_reference, axis_labels)
        """
        n_dim_numerical = numerical_coords.shape[1] if numerical_coords.ndim > 1 else 1

        has_y_ref = 'y_ref' in ref_solution
        has_t_ref = 't_ref' in ref_solution
        has_time_dim = has_y_ref or has_t_ref

        n_dim_reference = 2 if has_time_dim else 1

        if has_t_ref:
            axis_labels = ['x', 't']
        elif has_y_ref:
            y_label = ref_solution.get('y_label', 'y')
            axis_labels = ['x', y_label]
        else:
            axis_labels = ['x']

        return n_dim_numerical, n_dim_reference, axis_labels

    @staticmethod
    def prepare_comparison(
        numerical_coords: np.ndarray,
        numerical_values: np.ndarray,
        ref_solution: Dict,
        target_time: Optional[float] = None
    ) -> ComparisonPayload:
        """Prepare data for comparison by matching dimensions

        Args:
            numerical_coords: Numerical coordinates (n_points, n_dim)
            numerical_values: Numerical values (n_points,)
            ref_solution: Reference solution dictionary
            target_time: Target time for slicing (None = use max)

        Returns:
            ComparisonPayload with matched dimensions
        """
        n_dim_num, n_dim_ref, axis_labels = ReferenceSolutionHandler.detect_dimensions(
            ref_solution, numerical_coords
        )

        if n_dim_num == 1 and n_dim_ref == 1:
            return ReferenceSolutionHandler._prepare_1d_to_1d(
                numerical_coords, numerical_values, ref_solution, axis_labels
            )
        elif n_dim_num == 1 and n_dim_ref == 2:
            return ReferenceSolutionHandler._prepare_1d_to_2d(
                numerical_coords, numerical_values, ref_solution, axis_labels, target_time
            )
        elif n_dim_num == 2 and n_dim_ref == 2:
            return ReferenceSolutionHandler._prepare_2d_to_2d(
                numerical_coords, numerical_values, ref_solution, axis_labels
            )
        else:
            raise ValueError(f"Unsupported dimension combination: {n_dim_num}D numerical, {n_dim_ref}D reference")

    @staticmethod
    def _prepare_1d_to_1d(
        numerical_coords: np.ndarray,
        numerical_values: np.ndarray,
        ref_solution: Dict,
        axis_labels: list
    ) -> ComparisonPayload:
        """Prepare 1D to 1D comparison"""
        x_ref = ref_solution['x_ref']
        u_ref = ref_solution['u_ref']

        x_num = numerical_coords[:, 0] if numerical_coords.ndim > 1 else numerical_coords

        ref_coords = x_ref.reshape(-1, 1)
        num_coords = x_num.reshape(-1, 1)

        return ComparisonPayload(
            numerical_coords=num_coords,
            numerical_values=numerical_values.flatten(),
            reference_coords=ref_coords,
            reference_values=u_ref.flatten(),
            axis_labels=axis_labels,
            interpolation_method='cubic'
        )

    @staticmethod
    def _prepare_1d_to_2d(
        numerical_coords: np.ndarray,
        numerical_values: np.ndarray,
        ref_solution: Dict,
        axis_labels: list,
        target_time: Optional[float]
    ) -> ComparisonPayload:
        """Prepare 1D numerical to 2D reference comparison (with time slicing)"""
        x_ref = ref_solution['x_ref']
        y_ref = ref_solution.get('y_ref', None)
        if y_ref is None:
            y_ref = ref_solution.get('t_ref', None)
        if y_ref is None:
            raise ValueError("Reference solution must contain either 'y_ref' or 't_ref' for 1D-to-2D comparison")
        u_ref = ref_solution['u_ref']

        if target_time is None:
            target_time = np.max(y_ref)

        t_idx = np.argmin(np.abs(y_ref - target_time))
        t_actual = y_ref[t_idx]

        u_ref_grid = u_ref.reshape(len(y_ref), len(x_ref))
        u_ref_slice = u_ref_grid[t_idx, :]

        x_num = numerical_coords[:, 0] if numerical_coords.ndim > 1 else numerical_coords

        time_slice_info = {
            'target_time': target_time,
            'actual_time': t_actual,
            'time_index': t_idx,
            'axis_label': axis_labels[1]
        }

        return ComparisonPayload(
            numerical_coords=x_num.reshape(-1, 1),
            numerical_values=numerical_values.flatten(),
            reference_coords=x_ref.reshape(-1, 1),
            reference_values=u_ref_slice.flatten(),
            axis_labels=[axis_labels[0]],
            interpolation_method='cubic',
            time_slice_info=time_slice_info
        )

    @staticmethod
    def _prepare_2d_to_2d(
        numerical_coords: np.ndarray,
        numerical_values: np.ndarray,
        ref_solution: Dict,
        axis_labels: list
    ) -> ComparisonPayload:
        """Prepare 2D to 2D comparison"""
        x_ref = ref_solution['x_ref']
        y_ref = ref_solution.get('y_ref', None)
        if y_ref is None:
            y_ref = ref_solution.get('t_ref', None)
        if y_ref is None:
            raise ValueError("Reference solution must contain either 'y_ref' or 't_ref' for 2D-to-2D comparison")
        u_ref = ref_solution['u_ref']

        X_ref_grid, Y_ref_grid = np.meshgrid(x_ref, y_ref, indexing='xy')
        ref_coords = np.column_stack([X_ref_grid.flatten(), Y_ref_grid.flatten()])

        return ComparisonPayload(
            numerical_coords=numerical_coords,
            numerical_values=numerical_values.flatten(),
            reference_coords=ref_coords,
            reference_values=u_ref.flatten(),
            axis_labels=axis_labels,
            interpolation_method='linear'
        )

    @staticmethod
    def interpolate_to_points(
        source_coords: np.ndarray,
        source_values: np.ndarray,
        target_coords: np.ndarray,
        method: str = 'linear'
    ) -> np.ndarray:
        """Interpolate source data to target coordinates

        Args:
            source_coords: Source coordinates (n_source, n_dim)
            source_values: Source values (n_source,)
            target_coords: Target coordinates (n_target, n_dim)
            method: Interpolation method

        Returns:
            Interpolated values at target coordinates (n_target,)
        """
        if source_coords.shape[1] == 1:
            interp_func = interp1d(
                source_coords.flatten(),
                source_values.flatten(),
                kind=method if method != 'linear' else 'cubic',
                bounds_error=False,
                fill_value='extrapolate'
            )
            return interp_func(target_coords.flatten())
        else:
            result = griddata(
                source_coords,
                source_values.flatten(),
                target_coords,
                method=method
            )

            nan_mask = np.isnan(result)
            if np.any(nan_mask):
                result[nan_mask] = griddata(
                    source_coords,
                    source_values.flatten(),
                    target_coords[nan_mask],
                    method='nearest'
                )

            return result
