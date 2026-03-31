import numpy as np
from typing import Dict, Optional
from src.abstract_class.config.base_visualize import BaseVisualizer
from src.abstract_class.config.visualize_mixins import (
    ConvergenceMixin,
    ErrorAnalysisMixin,
    GridReconstructionMixin,
    LegacyAnalysisMixin,
    ReferenceComparisonMixin,
    SolutionPlotMixin,
)


class FuncFittingVisualizer(
    BaseVisualizer,
    SolutionPlotMixin,
    ReferenceComparisonMixin,
    ErrorAnalysisMixin,
    GridReconstructionMixin,
    ConvergenceMixin,
    LegacyAnalysisMixin,
):

    def plot_solution(self, data: Dict, prediction: np.ndarray, save_path: Optional[str] = None) -> None:
        x = data.get("x")
        u_true = data.get("u")
        if x is None and "x_segments" in data:
            x = np.vstack(data["x_segments"])
        if u_true is None:
            if "u" in data:
                u_true = data["u"]
            elif "u_segments" in data:
                u_true = np.vstack(data["u_segments"])
            elif "y_segments" in data:
                u_true = np.vstack(data["y_segments"])

        if x is None or u_true is None:
            raise ValueError("FuncFittingVisualizer.plot_solution requires x/u data")

        if self.n_dim == 1:
            self.plot_solution_1d(x, u_true, prediction, save_path, "Function Fitting Results")
        elif self.n_dim == 2:
            self.plot_solution_2d(x, u_true, prediction, save_path, "Function Fitting Results")
        else:
            raise ValueError(f"Unsupported dimensionality for plotting: {self.n_dim}")

    def _normalize_to_physical(self, x_comp: np.ndarray, segment_idx: int) -> np.ndarray:
        bounds = self.config.x_domain[segment_idx]
        return self.normalize_to_physical(x_comp, bounds)
