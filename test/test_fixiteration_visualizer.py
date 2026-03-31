from pathlib import Path

import numpy as np

from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.utils.visualize import (
    FixIterationPDEVisualizer,
)


class DummyConfig:
    def __init__(self):
        self.spatial_vars = ["x"]
        self.n_segments = [1]
        self.x_domain = [[0.0, 1.0]]
        self.realtime_visualization = False


def test_fixiteration_visualizer_animation_history_and_save(tmp_path):
    visualizer = FixIterationPDEVisualizer(DummyConfig())
    x = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
    u = np.linspace(1.0, 2.0, 5).reshape(-1, 1)

    visualizer.update_animation_data(2, 1e-2, u, {"x": x}, animation_skip=2)

    assert visualizer.iteration_history == [2]
    assert visualizer.residual_history == [1e-2]
    assert len(visualizer.solution_history) == 1

    save_dir = Path(tmp_path)
    visualizer.save_solution_data(
        U_train=u,
        U_test=u,
        coeffs_steady=np.ones((2, 1)),
        conv_info={"residual_history": [1.0, 0.5], "iterations": 2, "converged": True},
        data_train={"x": x},
        data_test={"x": x},
        save_dir=str(save_dir),
    )

    assert (save_dir / "solution_data.npz").exists()


def test_fixiteration_visualizer_plot_final_solution_creates_file(tmp_path):
    visualizer = FixIterationPDEVisualizer(DummyConfig())
    x = np.linspace(0.0, 1.0, 5).reshape(-1, 1)
    u = np.linspace(1.0, 2.0, 5).reshape(-1, 1)
    save_path = Path(tmp_path) / "final_solution.png"

    visualizer.plot_final_solution(u, {"x": x}, str(save_path))

    assert save_path.exists()
