import numpy as np
from typing import Dict, Optional
from src.abstract_class.config.base_visualize import BaseVisualizer
from src.abstract_class.config.visualize_mixins import SolutionPlotMixin, ConvergenceMixin


class LinearPDEVisualizer(BaseVisualizer, SolutionPlotMixin, ConvergenceMixin):

    def __init__(self, config=None):
        super().__init__(config)
        self.n_dim = config.n_dim if config else None

    def plot_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None, title: str = "PDE Solution"):
        x = data["x"]
        u_true = data["u"]
        u_pred = prediction

        if self.n_dim == 1:
            self.plot_solution_1d(x, u_true, u_pred, save_path, title)
        elif self.n_dim == 2:
            self.plot_solution_2d(x, u_true, u_pred, save_path, title)

    def plot_convergence(self, errors: list, save_path: Optional[str] = None, title: str = "Convergence History"):
        iterations = list(range(1, len(errors) + 1))
        self.plot_residual_history(errors, iterations, save_path)
