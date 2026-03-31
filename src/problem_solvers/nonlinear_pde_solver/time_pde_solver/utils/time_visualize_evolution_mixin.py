from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .time_visualize_basic_evolution_mixin import TimePDEBasicEvolutionMixin
from .time_visualize_reference_evolution_mixin import TimePDEReferenceEvolutionMixin


class TimePDEEvolutionMixin(
    TimePDEReferenceEvolutionMixin,
    TimePDEBasicEvolutionMixin,
):
    def plot_evolution_step(
        self,
        T: float,
        u: np.ndarray,
        x: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """绘制Time演化Process中的SingleTime step - According toDimensions自动选择Visualization方式"""
        handler = self._plot_evolution_step_1d if self.n_dim == 1 else self._plot_evolution_step_2d
        handler(T, u, x, save_path)

    def plot_evolution_step_with_reference(
        self,
        T: float,
        u: np.ndarray,
        x: np.ndarray,
        u_ref: Optional[np.ndarray] = None,
        solver=None,
        save_path: Optional[str] = None,
    ) -> None:
        """Unified evolution step plotting - eliminate dimension branching"""
        if self.n_dim == 1:
            self._plot_evolution_step_with_reference_1d(T, u, x, u_ref, solver, save_path)
            return
        self._plot_evolution_step_2d(T, u, x, save_path)

    def close_evolution_plot(self) -> None:
        """Shutdown time演化绘Graph"""
        plt.ioff()
        if hasattr(self, "fig"):
            plt.close(self.fig)

    def finalize_realtime_animation(self) -> None:
        """CompleteReal-timeanimationdisplay"""
        plt.ioff()
        print("Time evolution completed. Close the plot window to continue...")
