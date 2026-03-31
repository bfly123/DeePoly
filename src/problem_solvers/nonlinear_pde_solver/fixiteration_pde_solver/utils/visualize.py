from typing import Dict, Optional

import numpy as np

from src.abstract_class.config.base_visualize import BaseVisualizer
from src.abstract_class.config.visualize_mixins import (
    ConvergenceMixin,
    SolutionPlotMixin,
    GridReconstructionMixin,
    AnimationMixin,
    ReferenceComparisonMixin,
)
try:
    from src.abstract_class.config.visualize_mixins import RealtimeVisualizationMixin
except Exception:
    class RealtimeVisualizationMixin:  # type: ignore[override]
        """Fallback no-op mixin when optional realtime deps are unavailable."""

        pass

from src.visualization.dimensional_plotter import DimensionalVisualizer
from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.utils.fixiteration_visualize_output_mixin import (
    FixIterationVisualizeOutputMixin,
)
from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.utils.fixiteration_visualize_runtime_mixin import (
    FixIterationVisualizeRuntimeMixin,
)


class FixIterationPDEVisualizer(
    FixIterationVisualizeRuntimeMixin,
    FixIterationVisualizeOutputMixin,
    BaseVisualizer,
    ConvergenceMixin,
    SolutionPlotMixin,
    GridReconstructionMixin,
    AnimationMixin,
    RealtimeVisualizationMixin,
    ReferenceComparisonMixin,
):
    def __init__(self, config):
        super().__init__(config)
        self.dimensional_visualizer = DimensionalVisualizer(config)
        self.n_dim = self.dimensional_visualizer.n_dim
        self.iteration_history = []
        self.residual_history = []
        self.solution_history = []
        self._scatter_solution = None
        self._line_residual = None
        self._colorbar = None
        self.fig = None
        self.ax_solution = None
        self.ax_residual = None
        self._realtime_mode_active = False
