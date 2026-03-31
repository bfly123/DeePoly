import os

import numpy as np

from src.abstract_class.config.base_visualize import BaseVisualizer
from src.abstract_class.config.visualize_mixins import (
    AnimationMixin,
    ConvergenceMixin,
    DatasetVisualizationMixin,
    GridReconstructionMixin,
    ReferenceComparisonMixin,
)
from src.visualization.dimensional_plotter import DimensionalVisualizer

from .pseudotime_visualize_animation_mixin import PseudoTimeAnimationMixin
from .pseudotime_visualize_flow_mixin import PseudotimeVisualizeFlowMixin
from .pseudotime_visualize_output_mixin import PseudotimeVisualizeOutputMixin
from .pseudotime_visualize_reference_mixin import PseudoTimeReferenceMixin
from .pseudotime_visualize_render_mixin import PseudotimeVisualizeRenderMixin


class PseudoTimePDEVisualizer(
    PseudotimeVisualizeRenderMixin,
    PseudotimeVisualizeFlowMixin,
    PseudotimeVisualizeOutputMixin,
    PseudoTimeAnimationMixin,
    PseudoTimeReferenceMixin,
    BaseVisualizer,
    ConvergenceMixin,
    GridReconstructionMixin,
    ReferenceComparisonMixin,
    DatasetVisualizationMixin,
    AnimationMixin,
):
    """Pseudo-time PDE solver visualization built upon抽象化绘图组件。"""

    def __init__(self, config):
        super().__init__(config)
        self.dimensional_visualizer = DimensionalVisualizer(config)
        self.n_dim = self.dimensional_visualizer.n_dim
        self._init_animation_state()
