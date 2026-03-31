"""Dimension-aware plotting facade."""

from src.visualization.dimensional_plotter_reference_mixin import (
    DimensionalPlotterReferenceMixin,
)
from src.visualization.dimensional_plotter_solution_mixin import (
    DimensionalPlotterSolutionMixin,
)


class DimensionalVisualizer(
    DimensionalPlotterSolutionMixin,
    DimensionalPlotterReferenceMixin,
):
    """Plot solutions and references based on spatial dimensionality."""

    def __init__(self, config):
        self.config = config
        self.spatial_vars = getattr(config, "spatial_vars", ["x"])
        self.n_dim = len(self.spatial_vars)
