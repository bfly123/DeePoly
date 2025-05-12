import numpy as np
from typing import Dict
from abstract_class.config.base_data import BaseDataGenerator


class FuncFittingDataGenerator(BaseDataGenerator):
    """Data generator for function fitting problems"""

    def __init__(self, config):
        super().__init__(config)
        # Config is already saved in parent class

        self.custom_data_generator = self._load_custom_data_generator()

    def generate_global_field(self, x_global: np.ndarray) -> np.ndarray:
        """Generate global field, using sine function to create smooth distribution
        If a custom data generator exists, it will be used preferentially
        
        Args:
            x_global: Global point coordinates
            
        Returns:
            np.ndarray: Global field values
        """
        if self.custom_data_generator and hasattr(self.custom_data_generator, "generate_global_field"):
            return self.custom_data_generator.generate_global_field(x_global)
        else:
          raise ValueError("Custom data generator not found, or the custom data generator does not implement generate_global_field method")

    def generate_data(self, mode: str = "train") -> Dict:
        """Generate training/testing data
        If a custom data generator exists, it will be used preferentially
        """
        x_global = self._generate_global_points(mode)
        y_global = self.generate_global_field(x_global)

        # 2. Split into local segments
        x_segments,masks = self.split_global_points(x_global)
        y_segments = self.split_global_field(masks, y_global)

        # 3. Process segment data
        #x_swap, x_swap_norm, x_segments_norm, boundary_segments_dict = self._process_segments(x_segments)

        # 4. Prepare output data
        return self._prepare_output_dict(
            x_segments,
            y_segments,
            #x_segments_norm,
            #x_swap,
            #x_swap_norm,
            #boundary_segments_dict
        )
    def _prepare_output_dict(self, *args) -> Dict:
      """Prepare output data dictionary"""
      [
          x_segments,
          y_segments,
          #x_segments_norm,
          #x_swap,
          #x_swap_norm,
          #boundary_segments_dict,
      ] = args

      return {
          "x": np.vstack(x_segments),
          "u": np.vstack(y_segments),
          "x_min": self.config.x_min,
          "x_max": self.config.x_max,
          #"x_swap_norm": x_swap_norm,
          #"x_swap": x_swap,
          #"x_segments_norm": x_segments_norm,
          "x_segments": x_segments,
          "u_segments": y_segments,
      }
