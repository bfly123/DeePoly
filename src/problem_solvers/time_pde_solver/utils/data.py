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

        # Unified handling for all dimensions - read from config.Initial_conditions
        if self.config.Initial_conditions:
            ic = self.config.Initial_conditions[0]
            import sympy as sp

            # Create symbolic variables based on spatial_vars
            spatial_vars = getattr(self.config, 'spatial_vars', ['x'])
            n_dim = x_global.shape[1]

            # Create sympy symbols
            symbols = []
            symbol_dict = {'pi': sp.pi}
            for i, var_name in enumerate(spatial_vars[:n_dim]):
                sym = sp.Symbol(var_name)
                symbols.append(sym)
                symbol_dict[var_name] = sym

            # Parse initial condition expression
            expr = sp.sympify(ic['value'])

            # Create lambda function with appropriate number of arguments
            func = sp.lambdify(symbols, expr, 'numpy')

            # Evaluate initial condition
            if n_dim == 1:
                u_global[:, 0] = func(x_global[:, 0])
            else:
                # Multi-dimensional case: pass each coordinate column
                coords = [x_global[:, i] for i in range(n_dim)]
                u_global[:, 0] = func(*coords)

            # Diagnostic output
            print(f"Initial field generated from config: '{ic['value']}'")
            print(f"  Shape: {u_global.shape}, n_dim: {n_dim}")
            print(f"  Value range: [{u_global.min():.6e}, {u_global.max():.6e}]")
            print(f"  Mean: {u_global.mean():.6e}, Std: {u_global.std():.6e}")

        else:
            # Default initial conditions if not specified in config
            print("Warning: No Initial_conditions in config, using default cos(pi*x)")
            u_global[:, 0] = np.cos(np.pi * x_global[:, 0])

        return u_global

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