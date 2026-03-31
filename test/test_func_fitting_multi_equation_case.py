import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from src.problem_solvers.func_fitting_solver.core.net import FuncFittingNet
from src.problem_solvers.func_fitting_solver.solver import FuncFittingSolver
from src.problem_solvers.func_fitting_solver.utils import (
    FuncFittingConfig,
    FuncFittingDataGenerator,
)


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "cases" / "func_fitting_cases" / "multi_equation_2d_poly"


def _load_case_module(filename: str):
    module_path = CASE_DIR / filename
    spec = importlib.util.spec_from_file_location(f"case_module_{filename}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_func_fitting_net_physics_loss_uses_all_output_variables():
    class _Config:
        n_dim = 2
        hidden_dims = [4]
        n_eqs = 2
        device = "cpu"

    model = FuncFittingNet(_Config())
    x_train = torch.tensor([[0.0, 0.0], [0.5, -0.5]], dtype=torch.float64, requires_grad=True)
    u_train = torch.tensor([[1.0, -1.0], [0.5, 0.25]], dtype=torch.float64)

    loss = model.physics_loss({"x_train": x_train, "u_train": u_train})

    _, output = model(x_train)
    expected = torch.mean((output - u_train) ** 2)
    assert torch.allclose(loss, expected)


def test_func_fitting_model_init_accepts_config_object():
    config = FuncFittingConfig(case_dir=str(CASE_DIR))
    config.device = "cpu"

    model = FuncFittingNet.model_init(config)

    assert isinstance(model, FuncFittingNet)
    assert model.config is config


def test_multi_equation_2d_func_fitting_case_loads_and_generates_two_fields():
    config = FuncFittingConfig(case_dir=str(CASE_DIR))
    generator = FuncFittingDataGenerator(config)
    train_data = generator.generate_data("train")
    test_data = generator.generate_data("test")

    assert config.problem_type == "func_fitting"
    assert config.vars_list == ["u", "v"]
    assert config.n_eqs == 2
    assert train_data["u"].shape[1] == 2
    assert test_data["u"].shape[1] == 2

    module = _load_case_module("data_generate.py")
    expected = module.generate_reference_solution(test_data["x"])
    np.testing.assert_allclose(test_data["u"], expected)


def test_multi_equation_2d_func_fitting_case_solves_and_saves_outputs(tmp_path):
    config = FuncFittingConfig(case_dir=str(CASE_DIR))
    config.epochs_adam = 20
    config.epochs_lbfgs = 0
    config.max_retries = 0
    config.points_domain = [11, 11]
    config.points_domain_test = [11, 11]
    config.device = "cpu"
    config.linear_device = "cpu"

    solver = FuncFittingSolver(config=config, case_dir=str(CASE_DIR))
    train_predictions, test_predictions = solver.solve(result_dir=str(tmp_path))

    assert train_predictions.shape[1] == 2
    assert test_predictions.shape[1] == 2

    module = _load_case_module("data_generate.py")
    truth = module.generate_reference_solution(solver.data_test["x"])
    max_abs_error = float(np.max(np.abs(test_predictions - truth)))
    assert max_abs_error < 5e-2

    metrics_path = tmp_path / "metrics.json"
    solution_path = tmp_path / "solution_data.npz"
    u_plot_path = tmp_path / "u_comparison.png"
    v_plot_path = tmp_path / "v_comparison.png"
    assert metrics_path.exists()
    assert solution_path.exists()
    assert u_plot_path.exists()
    assert v_plot_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert set(metrics["variables"].keys()) == {"u", "v"}
    for var_name in ("u", "v"):
        assert metrics["variables"][var_name]["test_linf"] < 5e-2
