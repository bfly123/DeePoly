from src.algebraic_solver import FastNewtonSolver, LinearSolver
from src.visualization.dimensional_plotter import DimensionalVisualizer


class DummyConfig1D:
    spatial_vars = ["x"]


def test_linear_solver_cpu_only_init():
    solver = LinearSolver(use_gpu=False)
    assert solver.use_gpu is False


def test_fastnewton_cpu_only_init():
    solver = FastNewtonSolver(use_gpu=False)
    assert solver.use_gpu is False


def test_visualization_cpu_only_init():
    visualizer = DimensionalVisualizer(DummyConfig1D())
    info = visualizer.get_dimension_info()
    assert info["n_dim"] == 1
