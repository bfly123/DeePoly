import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import os
import importlib.util
from src.abstract_class.config.base_data import BaseDataGenerator


class TimePDEDataGenerator(BaseDataGenerator):
    """TimeRelatedPDEProblem的Data generator"""

    def __init__(self, config):
        super().__init__(config)
        """Initialize data generator
        
        Args:
            config: Configuration object
        """
        self.boundary_conditions = self.read_boundary_conditions()

    def generate_global_field(self, x_global: np.ndarray) -> np.ndarray:
        """Generate global field values - Initial conditions

        Args:
            x_global: Global point coordinates

        Returns:
            np.ndarray: Global field values (initial conditions)
        """
        u_global = np.zeros((x_global.shape[0], self.n_eqs))

        if self.config.Initial_conditions:
            for ic in self.config.Initial_conditions:
                var_name = ic.get("var", self.config.vars_list[0])
                value_expr = ic.get("value", "0")
                var_idx = self.config.vars_list.index(var_name)
                u_global[:, var_idx] = self._evaluate_initial_condition(value_expr, x_global)
                values = u_global[:, var_idx]
                print(f"Initial field generated for {var_name}: '{value_expr}'")
                print(f"  Range: [{values.min():.6e}, {values.max():.6e}]")
                print(f"  Mean: {values.mean():.6e}, Std: {values.std():.6e}")
        else:
            print("Warning: No Initial_conditions in config, using default cos(pi*x) for first variable")
            u_global[:, 0] = np.cos(np.pi * x_global[:, 0])

        return u_global

    def _evaluate_initial_condition(self, value_expr: str, x_global: np.ndarray) -> np.ndarray:
        import sympy as sp

        spatial_vars = getattr(self.config, "spatial_vars", ["x"])
        n_dim = x_global.shape[1]
        symbols = [sp.Symbol(var_name) for var_name in spatial_vars[:n_dim]]
        expr = sp.sympify(value_expr, locals={"pi": sp.pi})
        func = sp.lambdify(symbols, expr, "numpy")

        if n_dim == 1:
            values = func(x_global[:, 0])
        else:
            coords = [x_global[:, i] for i in range(n_dim)]
            values = func(*coords)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 0:
            values = np.full((x_global.shape[0],), float(values))
        return values.reshape(-1)

    def generate_data(self, mode: str = "train") -> Dict:
        """Generate training or test data

        Args:
            mode: Data mode, "train" or "test"

        Returns:
            data: Data dictionary containing x (coordinates) and U (values)
        """
        # Generate domain points
        x_global = self._generate_global_points(mode)
        initial_field = self.generate_global_field(x_global)  # Initial conditions场
        global_boundary_dict = self.read_boundary_conditions()
        x_segments, masks = self.split_global_points(x_global)
        initial_segments = self.split_global_field(masks, initial_field)
        x_segments_norm, x_swap, x_swap_norm, boundary_segments_dict = (
            self._process_segments(x_segments, global_boundary_dict)
        )

        return self._prepare_output_dict(
                x_segments,
                initial_segments,
                global_boundary_dict,
                x_segments_norm,
                x_swap,
                x_swap_norm,
                boundary_segments_dict,
        )

    def _prepare_output_dict(self, *args) -> Dict:
        """Prepare output data dictionary"""
        [
            x_segments,
            initial_segments,
            global_boundary_dict,
            x_segments_norm,
            x_swap,
            x_swap_norm,
            boundary_segments_dict,
        ] = args

        return {
            "x": np.vstack(x_segments),
            "x_segments": x_segments,
            "U": np.vstack(initial_segments),  # Initial conditionsData (兼容旧Interface)
            "U_seg": initial_segments,         # 分段Initial conditionsData (兼容旧Interface)
            "u_segments": initial_segments,    # Visualization兼容
            "initial": np.vstack(initial_segments),
            "initial_segments": initial_segments,
            "global_boundary_dict": global_boundary_dict,
            "x_segments_norm": x_segments_norm,
            "x_swap": x_swap,
            "x_swap_norm": x_swap_norm,
            "boundary_segments_dict": boundary_segments_dict,
        }
