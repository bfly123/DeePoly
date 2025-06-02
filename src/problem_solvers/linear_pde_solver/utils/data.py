import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import os
import importlib.util
from src.abstract_class.config.base_data import BaseDataGenerator
import matplotlib.pyplot as plt


class LinearPDEDataGenerator(BaseDataGenerator):
    """Linear partial differential equation data generator"""

    def __init__(self, config):
        super().__init__(config)
        """Initialize data generator
        
        Args:
            config: Configuration object
        """
        self.boundary_conditions = self.read_boundary_conditions()
        self.custom_data_generator = self._load_custom_data_generator()

    def generate_global_field(self, x_global: np.ndarray) -> np.ndarray:
        """Generate global field values

        Args:
            x_global: Global point coordinates

        Returns:
            np.ndarray: Global field values
        """
        return np.zeros((x_global.shape[0], self.n_eqs))

    def generate_data(self, mode: str = "train") -> Dict:
        """Generate training or test data

        Args:
            mode: Data mode, "train" or "test"

        Returns:
            data: Data dictionary containing x (coordinates) and u (values)
        """
        # Generate domain points
        x_global = self._generate_global_points(mode)
        source_term = self._load_source_term(x_global)
        #plot_source_term(x_global, source_term)
        global_boundary_dict = self.read_boundary_conditions()
        x_segments, masks = self.split_global_points(x_global)
        source_segments = self.split_global_field(masks, source_term)
        x_segments_norm, x_swap, x_swap_norm, boundary_segments_dict = (
            self._process_segments(x_segments, global_boundary_dict)
        )

        return self._prepare_output_dict(
                x_segments,
                source_segments,
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
            source_segments,
            global_boundary_dict,
            x_segments_norm,
            x_swap,
            x_swap_norm,
            boundary_segments_dict,
        ] = args

        return {
            "x": np.vstack(x_segments),
            "x_segments": x_segments,
            "source": np.vstack(source_segments),
            "source_segments": source_segments,
            "global_boundary_dict": global_boundary_dict,
            "x_segments_norm": x_segments_norm,
            "x_swap": x_swap,
            "x_swap_norm": x_swap_norm,
            "boundary_segments_dict": boundary_segments_dict,
        }

def plot_source_term(x_global, source_term):
    """
    Plot source term using scatter plot
    
    Args:
        x_global: Global point coordinates
        source_term: Source term values
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_global[:, 0], x_global[:, 1], 
                         c=source_term, 
                         cmap='coolwarm',
                         s=50)
    plt.colorbar(scatter, label='Source Term Value')
    plt.title('Source Term Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
