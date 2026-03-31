from pathlib import Path

import torch

from src.problem_solvers.nonlinear_pde_solver.time_pde_solver.core.net import TimePDENet
from src.problem_solvers.nonlinear_pde_solver.time_pde_solver.solver import TimePDESolver
from src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils.config import TimePDEConfig
from src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils.data import (
    TimePDEDataGenerator,
)


ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = (
    ROOT
    / "cases"
    / "nonlinear_pde_cases"
    / "time_pde_cases"
    / "AC_equation_imex_rk222_1d"
)


def test_time_pde_ac_case_config_and_data_load():
    config = TimePDEConfig(str(CASE_DIR))

    assert config.problem_type == "time_pde"
    assert config.time_scheme.lower() == "imex_rk_2_2_2"
    assert config.n_dim == 1
    assert config.n_eqs == 1
    assert list(config.n_segments) == [4]
    assert list(config.poly_degree) == [6]
    assert config.points_domain == [200]
    assert config.points_domain_test == [1024]

    datagen = TimePDEDataGenerator(config)
    train = datagen.generate_data("train")
    test = datagen.generate_data("test")

    assert train["x"].shape == (200, 1)
    assert train["U"].shape == (200, 1)
    assert test["x"].shape == (1024, 1)
    assert abs(train["U"].min() + 1.0) < 1e-10
    assert train["U"].max() > 0.05


def test_time_pde_ac_case_solver_initializes_reference_solution():
    config = TimePDEConfig(str(CASE_DIR))
    config.device = "cpu"
    config.linear_device = "cpu"
    solver = TimePDESolver(config=config, case_dir=str(CASE_DIR))

    assert solver.reference_solution is not None
    assert "x_ref" in solver.reference_solution
    assert "t_ref" in solver.reference_solution
    assert "u_ref" in solver.reference_solution
    assert solver.reference_solution["u_ref"].shape[0] == len(solver.reference_solution["t_ref"])


def test_time_pde_net_pads_empty_optional_operators():
    class DummyConfig:
        n_eqs = 1
        n_dim = 1
        hidden_dims = [4]

    net = TimePDENet(config=DummyConfig())
    template = torch.ones(5, dtype=torch.float64)
    padded = net.pad_operator_outputs(
        {"L1": [template], "L2": [], "F": [], "N": [template], "S": []}
    )

    assert len(padded["L1"]) == 1
    assert len(padded["L2"]) == 1
    assert len(padded["F"]) == 1
    assert len(padded["N"]) == 1
    assert len(padded["S"]) == 1
    assert torch.allclose(padded["L2"][0], torch.zeros_like(template))
