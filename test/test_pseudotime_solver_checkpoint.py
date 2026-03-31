from pathlib import Path

import torch

from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.solver import (
    PseudoTimePDESolver,
)


def test_pseudotime_solver_saves_spotter_model_checkpoint(tmp_path):
    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.model = torch.nn.Linear(2, 3)

    checkpoint_path = solver._save_spotter_model_checkpoint(str(tmp_path))

    assert checkpoint_path == str(tmp_path / "spotter_model.pth")
    assert Path(checkpoint_path).is_file()

    saved_state = torch.load(checkpoint_path, map_location="cpu")
    for key, value in solver.model.state_dict().items():
        assert key in saved_state
        assert torch.equal(saved_state[key], value.cpu())
