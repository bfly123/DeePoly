import argparse
import os
import sys

import matplotlib
import numpy as np
import scipy.io
import torch

try:
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..solver import NonlinearPDESolver
from .time_solver_factory_mixin import TimeSolverFactoryMixin
from .time_solver_postprocess_mixin import TimeSolverPostprocessMixin
from .time_solver_reference_mixin import TimeSolverReferenceMixin
from .time_solver_runtime_mixin import TimeSolverRuntimeMixin


class TimePDESolver(
    TimeSolverRuntimeMixin,
    TimeSolverReferenceMixin,
    TimeSolverPostprocessMixin,
    TimeSolverFactoryMixin,
    NonlinearPDESolver,
):
    """
    Optimized time PDE solver
    Supports IMEX-RK time stepping and hybrid neural network-polynomial methods
    """

    def _solver_specific_init(self) -> None:
        """Initialize solver-specific tracking once base components exist"""
        super()._solver_specific_init()
        self.loss_history = []
        if getattr(self, "visualizer", None) is not None:
            self.visualizer.reset_animation_data()

    def _should_train_network(self, iter: int) -> bool:
        """Determine whether to train the neural network at this iteration"""
        spotter_skip = getattr(self.config, "spotter_skip", 1)
        should_train = iter % spotter_skip == 0
        if not should_train and iter > 0:
            print(f"  Skipping neural network training (spotter_skip={spotter_skip}, it={iter})")
        return should_train

    def _on_network_trained(
        self,
        iter: int,
        dt: float,
        U_current: np.ndarray,
        final_loss: float,
        *,
        accepted: bool = True,
    ) -> None:
        """Record loss history and trigger optional NN prediction plots"""
        super()._on_network_trained(iter, dt, U_current, final_loss, accepted=accepted)
        self.loss_history.append(final_loss)
        if getattr(self.config, "plot_nn_predictions", False):
            self._plot_neural_network_predictions(iter, dt)


def solve_time_pde(case_dir: str) -> TimePDESolver:
    """便利function：直接SolveTimePDE"""
    solver = TimePDESolver(case_dir=case_dir)
    solver.solve()
    return solver
