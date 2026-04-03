import math
from collections import deque

import matplotlib.pyplot as plt

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

    @staticmethod
    def _default_max_frames(config) -> int:
        configured = getattr(config, "max_visualization_frames", None)
        if configured is not None:
            return int(configured)

        total_time = getattr(config, "T", None)
        dt = getattr(config, "dt", None)
        if total_time is None or dt in (None, 0):
            return 200

        estimated_steps = math.ceil(float(total_time) / float(dt))
        if str(getattr(config, "time_scheme", "")).lower() == "onestep_predictor":
            # The first predictor step uses a reduced dt, so one extra snapshot is expected.
            estimated_steps += 1
        return max(200, estimated_steps)

    def __init__(self, config):
        super().__init__(config)
        self.n_dim = len(config.spatial_vars) if hasattr(config, "spatial_vars") else len(config.x_domain)
        self._warned_realtime_backend = False
        self._realtime_window_closed = False
        self.fig = None
        self.ax = None

        self._frames = []
        self._save_gif = False
        self._visualization_interval = 1
        self._iteration_counter = 0
        self.cbar = None
        self._scatter_numerical = None
        self._line_reference = None
        self._step_counter = 0
        self._max_frames = self._default_max_frames(config)
        self.time_history_viz = deque(maxlen=self._max_frames)
        self.solution_history_viz = deque(maxlen=self._max_frames)
        self.coeffs_history_viz = deque(maxlen=self._max_frames)

        if self._realtime_requested():
            self._initialize_realtime_figure()

    def _realtime_requested(self) -> bool:
        return bool(getattr(self.config, "realtime_visualization", False))

    def _backend_supports_realtime(self) -> bool:
        backend = plt.get_backend().lower()
        noninteractive = {
            "agg",
            "cairo",
            "pdf",
            "pgf",
            "ps",
            "svg",
            "template",
        }
        if backend in noninteractive:
            return False
        if backend.startswith("module://matplotlib_inline"):
            return False
        return True

    def _initialize_realtime_figure(self) -> bool:
        if self._realtime_window_closed:
            return False
        if not self._backend_supports_realtime():
            if not self._warned_realtime_backend:
                print(
                    "Realtime visualization is enabled, but the current Matplotlib backend "
                    f"'{plt.get_backend()}' is non-interactive. Skipping live window updates."
                )
                self._warned_realtime_backend = True
            return False
        if self.fig is not None and self.ax is not None:
            return True

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

        def _mark_closed(_event) -> None:
            self._realtime_window_closed = True
            self.fig = None
            self.ax = None

        self.fig.canvas.mpl_connect("close_event", _mark_closed)
        try:
            manager = getattr(self.fig.canvas, "manager", None)
            if manager is not None and hasattr(manager, "show"):
                manager.show()
            else:
                self.fig.show()
        except Exception:
            pass
        plt.pause(0.001)
        return True

    def _ensure_realtime_figure(self) -> bool:
        if not self._realtime_requested():
            return False
        if self._realtime_window_closed:
            return False
        if self.fig is None or self.ax is None:
            return self._initialize_realtime_figure()
        if not plt.fignum_exists(self.fig.number):
            self._realtime_window_closed = True
            self.fig = None
            self.ax = None
            print("Realtime visualization window was closed. Disabling further live updates.")
            return False
        return True
