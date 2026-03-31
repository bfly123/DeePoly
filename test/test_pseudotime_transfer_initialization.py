from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.solver import (
    PseudoTimePDESolver,
)


def test_transfer_initialization_loads_weights_and_training_field(tmp_path):
    source_case = tmp_path / "lid_driven_cavity_re100"
    source_results = source_case / "results"
    source_results.mkdir(parents=True)
    (source_case / "config.json").write_text("{}", encoding="utf-8")

    source_model = torch.nn.Linear(2, 3)
    with torch.no_grad():
        source_model.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ],
                dtype=source_model.weight.dtype,
            )
        )
        source_model.bias.copy_(
            torch.tensor([0.1, 0.2, 0.3], dtype=source_model.bias.dtype)
        )
    torch.save(source_model.state_dict(), source_results / "spotter_model.pth")

    x_train = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        dtype=np.float64,
    )
    transferred_field = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )
    np.savez(
        source_results / "solution_data.npz",
        x_train=x_train,
        sniper_train=transferred_field,
    )

    target_case = tmp_path / "lid_driven_cavity_re1000"
    target_case.mkdir()
    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.case_dir = str(target_case)
    solver.config = SimpleNamespace(
        case_dir=str(target_case),
        device="cpu",
        n_eqs=3,
        initial_condition={
            "method": "transfer",
            "transfer": {
                "source_case": "../lid_driven_cavity_re100",
                "solution_key": "sniper_train",
            },
        },
    )
    solver.data_train = {"x": x_train.copy()}
    solver.model = torch.nn.Linear(2, 3)

    U_init = solver._transfer_initialization()

    np.testing.assert_allclose(U_init, transferred_field)
    loaded_state = solver.model.state_dict()
    for key, value in source_model.state_dict().items():
        assert key in loaded_state
        assert torch.equal(loaded_state[key], value)


def test_transfer_initialization_interpolates_when_train_points_do_not_match(tmp_path):
    source_results = tmp_path / "source_results"
    source_results.mkdir(parents=True)

    source_x = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    source_u = np.column_stack(
        [
            source_x[:, 0] + source_x[:, 1],
            2.0 * source_x[:, 0] - source_x[:, 1],
        ]
    )
    np.savez(
        source_results / "solution_data.npz",
        x_train=source_x,
        sniper_train=source_u,
    )

    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.case_dir = str(tmp_path)
    solver.config = SimpleNamespace(
        case_dir=str(tmp_path),
        device="cpu",
        n_eqs=2,
        initial_condition={
            "method": "transfer",
            "transfer": {
                "results_dir": str(source_results),
                "load_network_weights": False,
                "solution_key": "sniper_train",
            },
        },
    )
    solver.data_train = {
        "x": np.array(
            [
                [0.5, 0.5],
                [0.25, 0.75],
            ],
            dtype=np.float64,
        )
    }
    solver.model = torch.nn.Linear(2, 2)

    U_init = solver._transfer_initialization()

    expected = np.column_stack(
        [
            solver.data_train["x"][:, 0] + solver.data_train["x"][:, 1],
            2.0 * solver.data_train["x"][:, 0] - solver.data_train["x"][:, 1],
        ]
    )
    np.testing.assert_allclose(U_init, expected, atol=1e-10, rtol=1e-10)


def test_transfer_initialization_enables_skip_initial_training_only_with_loaded_weights(tmp_path):
    source_case = tmp_path / "source_case"
    source_results = source_case / "results"
    source_results.mkdir(parents=True)
    (source_case / "config.json").write_text("{}", encoding="utf-8")

    source_model = torch.nn.Linear(2, 3)
    torch.save(source_model.state_dict(), source_results / "spotter_model.pth")

    x_train = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    transferred_field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    np.savez(source_results / "solution_data.npz", x_train=x_train, sniper_train=transferred_field)

    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.case_dir = str(tmp_path / "target_case")
    solver.config = SimpleNamespace(
        case_dir=solver.case_dir,
        device="cpu",
        n_eqs=3,
        initial_condition={
            "method": "transfer",
            "transfer": {
                "source_case": str(source_case),
                "solution_key": "sniper_train",
                "skip_initial_training": True,
            },
        },
    )
    solver.data_train = {"x": x_train.copy()}
    solver.model = torch.nn.Linear(2, 3)

    U_init = solver._transfer_initialization()

    np.testing.assert_allclose(U_init, transferred_field)
    assert solver._transferred_network_weights_loaded is True
    assert solver._skip_initial_spotter_training is True


def test_transfer_initialization_loads_coefficients_when_requested(tmp_path):
    source_case = tmp_path / "source_case"
    source_results = source_case / "results"
    source_results.mkdir(parents=True)
    (source_case / "config.json").write_text("{}", encoding="utf-8")

    x_train = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    transferred_field = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    transferred_coeffs = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    np.savez(
        source_results / "solution_data.npz",
        x_train=x_train,
        sniper_train=transferred_field,
        coeffs=transferred_coeffs,
    )

    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.case_dir = str(tmp_path / "target_case")
    solver.config = SimpleNamespace(
        case_dir=solver.case_dir,
        device="cpu",
        n_eqs=2,
        initial_condition={
            "method": "transfer",
            "transfer": {
                "source_case": str(source_case),
                "load_network_weights": False,
                "load_coefficients": True,
                "solution_key": "sniper_train",
            },
        },
    )
    solver.data_train = {"x": x_train.copy()}
    solver.model = torch.nn.Linear(2, 2)

    U_init = solver._transfer_initialization()

    np.testing.assert_allclose(U_init, transferred_field)
    np.testing.assert_allclose(solver._transferred_coeffs_init, transferred_coeffs)


def test_jaxpi_transfer_initialization_loads_exported_field(tmp_path):
    source_results = tmp_path / "jaxpi_results"
    source_results.mkdir(parents=True)

    x_train = np.array(
        [
            [-1.0, 0.0],
            [0.0, 0.5],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    transferred_field = np.array(
        [
            [-0.5],
            [0.25],
            [0.75],
        ],
        dtype=np.float64,
    )
    np.savez(
        source_results / "jaxpi_transfer.npz",
        x_train=x_train,
        U_train=transferred_field,
    )

    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.case_dir = str(tmp_path / "target_case")
    solver.config = SimpleNamespace(
        case_dir=solver.case_dir,
        device="cpu",
        n_eqs=1,
        initial_condition={
            "method": "jaxpi_transfer",
            "jaxpi_transfer": {
                "results_dir": str(source_results),
                "teacher_fit_epochs": 0,
            },
        },
    )
    solver.data_train = {"x": x_train.copy()}
    solver.model = torch.nn.Linear(2, 1)

    U_init = solver._jaxpi_transfer_initialization()

    np.testing.assert_allclose(U_init, transferred_field)
