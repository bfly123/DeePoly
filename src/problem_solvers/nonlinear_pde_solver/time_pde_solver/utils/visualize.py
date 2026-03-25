import matplotlib.pyplot as plt
from collections import deque

from src.abstract_class.config.base_visualize import BaseVisualizer

from .time_visualize_animation_mixin import TimePDEAnimationMixin
from .time_visualize_evolution_mixin import TimePDEEvolutionMixin
from .time_visualize_mesh_mixin import TimePDEMeshMixin
from .time_visualize_solution_mixin import TimePDESolutionMixin
from .time_visualize_summary_mixin import TimePDESummaryMixin


class TimePDEVisualizer(
    TimePDEAnimationMixin,
    TimePDEEvolutionMixin,
    TimePDEMeshMixin,
    TimePDESolutionMixin,
    TimePDESummaryMixin,
    BaseVisualizer,
):
    """Visualizer for time-dependent PDE problems - supports independent 1D and 2D visualization"""

    def __init__(self, config):
        super().__init__(config)
        self.n_dim = len(config.spatial_vars) if hasattr(config, "spatial_vars") else len(config.x_domain)
        plt.ion()

        figure_config = {
            1: {"figsize": (6, 4), "projection": None},
            2: {"figsize": (6, 6), "projection": None},
        }
        config_2d = figure_config.get(self.n_dim, figure_config[1])
        self.fig = plt.figure(figsize=config_2d["figsize"])
        if config_2d["projection"]:
            self.ax = self.fig.add_subplot(111, projection=config_2d["projection"])
        else:
            self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=1.0)

        self._frames = []
        self._save_gif = False
        self._visualization_interval = 1
        self._iteration_counter = 0
        self.cbar = None
        self._scatter_numerical = None
        self._line_reference = None
        self._step_counter = 0
        self._max_frames = getattr(config, "max_visualization_frames", 200)
        self.time_history_viz = deque(maxlen=self._max_frames)
        self.solution_history_viz = deque(maxlen=self._max_frames)
        self.coeffs_history_viz = deque(maxlen=self._max_frames)
