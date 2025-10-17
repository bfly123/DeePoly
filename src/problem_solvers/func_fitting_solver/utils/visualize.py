import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from src.abstract_class.config.base_visualize import BaseVisualizer


class FuncFittingVisualizer(BaseVisualizer):

    def plot_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        fig = self._create_figure(figsize=(12, 8))
        x_segments = data["x_segments"]
        y_segments = data["y_segments"]

        start_idx = 0
        for i in range(self.Ns):
            segment_size = x_segments[i].shape[0]
            end_idx = start_idx + segment_size
            x_phys = self._normalize_to_physical(x_segments[i], i)

            plt.plot(x_phys[:, 0], y_segments[i][:, 0], 'b.', label='True' if i == 0 else None)
            plt.plot(x_phys[:, 0], prediction[start_idx:end_idx, 0], 'r-', label='Prediction' if i == 0 else None)
            start_idx = end_idx

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Function Fitting Results')
        plt.legend()
        plt.grid(True)

        if save_path:
            self._save_figure(fig, save_path)
        self._close_figure(fig)
