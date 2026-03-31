from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils.visualize import TimePDEVisualizer


pytestmark = pytest.mark.filterwarnings("ignore:Glyph .* missing from font")


class DummyTimeConfig:
    def __init__(self, n_dim=1):
        self.spatial_vars = ["x"] if n_dim == 1 else ["x", "y"]
        self.x_domain = [[0.0, 1.0]] if n_dim == 1 else [[0.0, 1.0], [0.0, 1.0]]
        self.n_segments = [2] if n_dim == 1 else [2, 2]
        self.n_eqs = 1
        self.max_visualization_frames = 5
        self.realtime_visualization = False
        self.animation_skip = 1
        self.T = 1.0
        self.points_domain_test = [4] if n_dim == 1 else [4, 4]
        self.points_per_segment_test = [2, 2]
        self.device = "cpu"
        self.x_min = [np.array([0.0, 0.0]) for _ in range(4)]
        self.x_max = [np.array([0.5, 0.5]), np.array([1.0, 0.5]), np.array([0.5, 1.0]), np.array([1.0, 1.0])]


def test_time_pde_visualizer_plot_solution_1d(tmp_path):
    visualizer = TimePDEVisualizer(DummyTimeConfig(n_dim=1))
    x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    data = {
        "x_segments": [x[:4], x[4:]],
        "u_segments": [np.sin(x[:4]), np.sin(x[4:])],
    }
    prediction = np.sin(x)
    save_path = tmp_path / "solution_1d.png"
    visualizer.plot_solution(data, prediction, str(save_path))
    assert save_path.exists()


def test_time_pde_visualizer_plot_solution_2d_and_reset_animation(tmp_path):
    visualizer = TimePDEVisualizer(DummyTimeConfig(n_dim=2))
    xs = np.linspace(0.0, 1.0, 4)
    ys = np.linspace(0.0, 1.0, 4)
    X, Y = np.meshgrid(xs, ys)
    coords = np.column_stack((X.ravel(), Y.ravel()))
    values = (coords[:, :1] + coords[:, 1:2])
    data = {"x_segments": [coords], "u_segments": [values]}
    save_path = tmp_path / "solution_2d.png"
    visualizer.plot_solution(data, values, str(save_path))
    assert save_path.exists()

    visualizer.time_history_viz.append(0.1)
    visualizer.solution_history_viz.append(values)
    visualizer.reset_animation_data()
    time_history, solution_history = visualizer.get_animation_data()
    assert time_history == []
    assert solution_history == []


def test_time_pde_visualizer_plot_evolution_with_reference_1d(tmp_path):
    class DummySolver:
        reference_solution = {"x_ref": np.linspace(0.0, 1.0, 8)}

    visualizer = TimePDEVisualizer(DummyTimeConfig(n_dim=1))
    x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    u = np.sin(x)
    u_ref = np.sin(DummySolver.reference_solution["x_ref"])
    save_path = tmp_path / "evolution_ref_1d.png"

    visualizer.plot_evolution_step_with_reference(
        0.1,
        u,
        x,
        u_ref=u_ref,
        solver=DummySolver(),
        save_path=str(save_path),
    )

    assert save_path.exists()
    assert visualizer._scatter_numerical is not None
