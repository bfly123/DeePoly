import sympy as sp

from src.meta_coding.auto_spotter import AutoCodeGenerator, EquationProcessor, update_pytorch_net_code


def test_parse_equation_converts_equality_and_derivatives():
    processor = EquationProcessor(["x"], ["u"])
    expr = processor.parse_equation("diff(u,x,2) = u")
    assert expr == sp.sympify("u_xx - u")


def test_find_used_derivatives_includes_lower_orders():
    processor = EquationProcessor(["x"], ["u"])
    used = processor._find_used_derivatives(["diff(u,x,3) + u"])
    assert used["u"] == 0
    assert used["u_x"] == 1
    assert used["u_xx"] == 2
    assert used["u_xxx"] == 3


def test_generate_pytorch_derivatives_emits_expected_lines():
    processor = EquationProcessor(["x", "y"], ["u"])
    code = processor.generate_pytorch_derivatives(["diff(u,x) + diff(u,y,2)"])
    assert "u = U[..., 0]" in code
    assert "du_x = self.gradients(u, x_train)[0][..., 0]" in code
    assert "du_yy = self.gradients(du_y, x_train)[0][..., 1]" in code


def test_convert_equation_to_pytorch_rewrites_derivatives():
    processor = EquationProcessor(["x", "y"], ["u"])
    converted = processor._convert_equation_to_pytorch("diff(u,x) + diff(u,y,3)")
    assert converted == "du_x + du_yyy"


def test_auto_code_generator_consistency_detects_matching_generated_block(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "problem_type": "linear_pde",
          "spatial_vars": ["x"],
          "vars_list": ["u"],
          "eq": {"L1": ["diff(u,x,2) + u"]}
        }
        """.strip()
    )
    net_file = tmp_path / "base_net.py"
    net_file.write_text("# auto code begin\n# auto code end\n")

    generator = AutoCodeGenerator(str(config_path))
    generator._generate_and_update_code(str(net_file))

    needs_regen, reason = generator.check_config_net_consistency(str(net_file))
    assert needs_regen is False
    assert "一致" in reason


def test_auto_code_generator_consistency_detects_signature_change(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "problem_type": "linear_pde",
          "spatial_vars": ["x"],
          "vars_list": ["u"],
          "eq": {"L1": ["diff(u,x,2) + u"]}
        }
        """.strip()
    )
    net_file = tmp_path / "base_net.py"
    net_file.write_text("# auto code begin\n# auto code end\n")

    generator = AutoCodeGenerator(str(config_path))
    generator._generate_and_update_code(str(net_file))

    config_path.write_text(
        """
        {
          "problem_type": "linear_pde",
          "spatial_vars": ["x"],
          "vars_list": ["u"],
          "eq": {"L1": ["diff(u,x,2) - u"]}
        }
        """.strip()
    )
    changed_generator = AutoCodeGenerator(str(config_path))
    needs_regen, reason = changed_generator.check_config_net_consistency(str(net_file))
    assert needs_regen is True
    assert "更改" in reason


def test_update_pytorch_net_code_replaces_auto_code_section(tmp_path):
    net_file = tmp_path / "base_net.py"
    net_file.write_text("before\n# auto code begin\nold\n# auto code end\nafter\n")

    update_pytorch_net_code(
        equations=("diff(u,x)",),
        vars_list=["u"],
        dimensions=["x"],
        net_file_path=str(net_file),
    )

    content = net_file.read_text()
    assert "du_x = self.gradients(u, x_train)[0][..., 0]" in content
    assert "before" in content
    assert "after" in content
