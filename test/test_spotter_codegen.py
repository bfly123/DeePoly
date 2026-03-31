import json

from src.meta_coding.spotter_codegen import compose_pytorch_code, extract_saved_signature
from src.meta_coding.spotter_equation_processor import EquationProcessor


def _build_config():
    return {
        "problem_type": "general_pde",
        "vars_list": ["u", "v", "p"],
        "spatial_vars": ["x", "y"],
        "const_list": [{"Re": 100, "nu": 0.01}],
        "eq": {
            "L1": [
                "diff(u,x) + diff(v,y)",
                "diff(p,x) - nu*diff(u,x,2)",
                "diff(p,y) - nu*diff(v,y,2)",
            ],
            "L2": ["0", "0", "0"],
            "F": ["0", "0", "0"],
            "N": ["0", "u*diff(u,x)", "v*diff(v,y)"],
            "S": ["0", "x", "y"],
        },
    }


def test_extract_saved_signature_supports_nested_const_list_json():
    config = _build_config()
    processor = EquationProcessor(config["spatial_vars"], config["vars_list"])

    generated = compose_pytorch_code(
        processor=processor,
        equations=[],
        problem_type=config["problem_type"],
        config_dict=config,
    )

    signature = extract_saved_signature(generated)

    assert signature is not None
    assert signature["const_list"] == config["const_list"]
    assert signature["S"] == config["eq"]["S"]


def test_compose_pytorch_code_emits_constants_and_source_operator_blocks():
    config = _build_config()
    processor = EquationProcessor(config["spatial_vars"], config["vars_list"])

    generated = compose_pytorch_code(
        processor=processor,
        equations=[],
        problem_type=config["problem_type"],
        config_dict=config,
    )

    assert "# Extract configured constants" in generated
    assert "nu = 0.01" in generated
    assert "Re = 100" in generated
    assert "# S operators" in generated
    assert "S = [" in generated

    signature_line = next(
        line for line in generated.splitlines() if line.startswith("# Config signature:")
    )
    parsed_signature = json.loads(signature_line.split(":", 1)[1].strip())
    assert parsed_signature["const_list"] == config["const_list"]


def test_compose_pytorch_code_preserves_time_like_derivative_for_allen_cahn_case():
    config = {
        "problem_type": "fixiteration_pde",
        "vars_list": ["u"],
        "spatial_vars": ["x", "y"],
        "eq": {
            "L1": ["-0.0001*diff(u,x,2)+diff(u,y)"],
            "L2": ["0"],
            "F": ["0"],
            "N": ["-5*u+5*u**3"],
            "S": ["0"],
        },
    }
    processor = EquationProcessor(config["spatial_vars"], config["vars_list"])

    generated = compose_pytorch_code(
        processor=processor,
        equations=[],
        problem_type=config["problem_type"],
        config_dict=config,
    )

    assert "du_y = self.gradients(u, x_train)[0][..., 1]" in generated
    assert "L1 = [-0.0001*du_xx+du_y]" in generated
