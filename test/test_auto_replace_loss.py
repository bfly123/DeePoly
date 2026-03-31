import json

from src.problem_solvers.linear_pde_solver.auto_replace_loss import (
    LossCodeGenerator,
    update_config_auto_code,
)


def test_loss_code_generator_builds_derivative_and_equation_code():
    generator = LossCodeGenerator(
        spatial_vars=["x", "y"],
        vars_list=["u"],
        const_list=[{"Re": 100}],
        model_path="unused.py",
    )

    derivatives_code = generator._generate_derivatives_code(
        ["diff(u,x,2) + diff(u,y)"]
    )
    equations_code = generator._generate_equations_code(
        ["diff(u,x,2) + diff(u,y)"]
    )

    assert "Re = 100" in derivatives_code
    assert "du_x = self.gradients(u, x_train)[0][..., 0]" in derivatives_code
    assert "du_xx = self.gradients(du_x, x_train)[0][..., 0]" in derivatives_code
    assert "eq = [du_xx + du_y]" in equations_code


def test_loss_code_generator_updates_auto_code_block(tmp_path):
    model_file = tmp_path / "net.py"
    model_file.write_text(
        "def physics_loss(self):\n"
        "    # auto code begin\n"
        "    old = True\n"
        "    # auto code end\n"
        "    return pde_loss\n"
    )
    generator = LossCodeGenerator(
        spatial_vars=["x"],
        vars_list=["u"],
        model_path=str(model_file),
    )

    generator.update_code(["diff(u,x,2)"], ["u**2"])

    updated = model_file.read_text()
    backup = model_file.with_suffix(model_file.suffix + ".backup")
    assert backup.exists()
    assert "du_xx = self.gradients(du_x, x_train)[0][..., 0]" in updated
    assert "eq[0] = eq[0] + u**2" in updated
    assert "pde_loss = torch.mean((eq[0] - source[0]) ** 2)" in updated
    assert "old = True" not in updated


def test_update_config_auto_code_toggles_value(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"auto_code": True, "name": "demo"}))

    update_config_auto_code(str(config_path), False)

    config = json.loads(config_path.read_text())
    assert config["auto_code"] is False
    assert config["name"] == "demo"
