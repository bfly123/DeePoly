from pathlib import Path

import numpy as np
import scipy.io

from src.abstract_class.config.base_data import BaseDataGenerator
from src.abstract_class.config.data_helpers import (
    CustomDataModuleLoader,
    MathExpressionEvaluator,
    ReferenceSolutionLoader,
    SourceTermResolver,
)


class DummyConfig:
    def __init__(self, case_dir=None):
        self.n_segments = [1, 1]
        self.points_per_swap = 1
        self.n_dim = 2
        self.n_eqs = 1
        self.case_dir = case_dir
        self.spatial_vars = ["x", "y"]
        self.vars_list = ["u"]
        self.eq = {}
        self.source_term = None
        self.reference_solution = None
        self.x_domain = [[0.0, 1.0], [0.0, 2.0]]
        self.x_min = [np.array([0.0, 0.0])]
        self.x_max = [np.array([1.0, 2.0])]
        self.boundary_conditions = []


class DummyDataGenerator(BaseDataGenerator):
    def generate_global_field(self, x_global: np.ndarray) -> np.ndarray:
        return np.zeros((x_global.shape[0], self.n_eqs))

    def generate_data(self, mode: str = "train"):
        return {}

    def _prepare_output_dict(self, *args):
        return {}


def test_math_expression_evaluator_constant():
    evaluator = MathExpressionEvaluator(["x", "y"])
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    result = evaluator.evaluate("2.5", x)
    assert result.shape == (2, 1)
    assert np.allclose(result, 2.5)


def test_math_expression_evaluator_multivar_expression():
    evaluator = MathExpressionEvaluator(["x", "y"])
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    result = evaluator.evaluate("x + 2*y", x)
    assert np.allclose(result[:, 0], [2.0, 8.0])


def test_math_expression_evaluator_invalid_expression_returns_zeros():
    evaluator = MathExpressionEvaluator(["x", "y"])
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    result = evaluator.evaluate("bad_func(x)", x)
    assert result.shape == (2, 1)
    assert np.allclose(result, 0.0)


def test_piecewise_expression_evaluator():
    evaluator = MathExpressionEvaluator(["x", "y"])
    x = np.array([[-1.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
    piecewise = {
        "type": "piecewise",
        "variable": "x",
        "pieces": [
            {"range": [-10.0, 0.0], "expression": "-1"},
            {"range": [0.0, 1.0], "expression": "x + 1"},
        ],
    }
    result = evaluator.evaluate_piecewise(piecewise, x)
    assert np.allclose(result[:, 0], [-1.0, 1.5, 0.0])


def test_custom_data_module_loader_without_sys_path_mutation(tmp_path):
    module_path = tmp_path / "data_generate.py"
    module_path.write_text(
        "def generate_source_term(x):\n"
        "    import numpy as np\n"
        "    return np.sum(x, axis=1)\n"
    )
    loader = CustomDataModuleLoader()
    module = loader.load_from_case_dir(str(tmp_path))
    values = module.generate_source_term(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.allclose(values, [3.0, 7.0])


def test_base_data_generator_uses_custom_loader(tmp_path):
    module_path = tmp_path / "data_generate.py"
    module_path.write_text(
        "def generate_source_term(x):\n"
        "    import numpy as np\n"
        "    return np.ones((x.shape[0], 1))\n"
    )
    generator = DummyDataGenerator(DummyConfig(case_dir=str(tmp_path)))
    assert generator.custom_data_generator is not None


def test_base_data_generator_expression_parser_delegates():
    generator = DummyDataGenerator(DummyConfig())
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    result = generator._parse_math_expression("x + y", x)
    assert np.allclose(result[:, 0], [1.0, 5.0])


def test_source_term_resolver_string_expression():
    config = DummyConfig()
    config.eq = {"S": "x + y"}
    resolver = SourceTermResolver(MathExpressionEvaluator(config.spatial_vars))
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    result = resolver.resolve(config, None, x, 1)
    assert np.allclose(result[:, 0], [1.0, 5.0])


def test_source_term_resolver_list_expression():
    config = DummyConfig()
    config.n_eqs = 2
    config.eq = {"S": ["x", "y"]}
    resolver = SourceTermResolver(MathExpressionEvaluator(config.spatial_vars))
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = resolver.resolve(config, None, x, 2)
    assert np.allclose(result, [[1.0, 2.0], [3.0, 4.0]])


def test_source_term_resolver_custom_module(tmp_path):
    module_path = tmp_path / "data_generate.py"
    module_path.write_text(
        "def generate_source_term(x):\n"
        "    import numpy as np\n"
        "    return np.sum(x, axis=1)\n"
    )
    config = DummyConfig(case_dir=str(tmp_path))
    config.eq = {"S": "data_generate.py"}
    generator = DummyDataGenerator(config)
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = generator._load_source_term(x)
    assert np.allclose(result[:, 0], [3.0, 7.0])


def test_reference_solution_loader_expression():
    config = DummyConfig()
    config.reference_solution = "x - y"
    resolver = ReferenceSolutionLoader(MathExpressionEvaluator(config.spatial_vars))
    x = np.array([[1.0, 2.0], [5.0, 3.0]])
    result = resolver.load(config, None, x, 1)
    assert np.allclose(result[:, 0], [-1.0, 2.0])


def test_reference_solution_loader_from_custom_module(tmp_path):
    module_path = tmp_path / "data_generate.py"
    module_path.write_text(
        "def generate_reference_solution(x):\n"
        "    import numpy as np\n"
        "    return np.prod(x, axis=1)\n"
    )
    config = DummyConfig(case_dir=str(tmp_path))
    config.reference_solution = "data_generate.py"
    generator = DummyDataGenerator(config)
    x = np.array([[2.0, 3.0], [4.0, 5.0]])
    result = generator.load_reference_solution(x)
    assert np.allclose(result[:, 0], [6.0, 20.0])


def test_reference_solution_loader_from_mat_file(tmp_path):
    mat_path = tmp_path / "reference.mat"
    scipy.io.savemat(mat_path, {"u": np.array([[1.0], [2.0], [3.0]])})
    config = DummyConfig(case_dir=str(tmp_path))
    config.reference_solution = "reference.mat"
    generator = DummyDataGenerator(config)
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    result = generator.load_reference_solution(x)
    assert np.allclose(result[:, 0], [1.0, 2.0, 3.0])


def test_generate_boundary_points_for_dim_region():
    generator = DummyDataGenerator(DummyConfig())
    points = generator._generate_boundary_points("dim0_max", 3)
    assert points.shape == (3, 2)
    assert np.allclose(points[:, 0], 1.0)
    assert np.allclose(points[:, 1], [0.0, 1.0, 2.0])


def test_region_to_segment_and_normals_for_named_boundary():
    generator = DummyDataGenerator(DummyConfig())
    assert generator._region_to_segment("left") == 0
    normals = generator._get_boundary_normals("top", 2)
    assert np.allclose(normals, np.array([[0.0, 1.0], [0.0, 1.0]]))


def test_read_boundary_conditions_and_project_segment():
    config = DummyConfig()
    config.boundary_conditions = [
        {"type": "dirichlet", "region": "left", "points": 3, "value": "x + 1"}
    ]
    generator = DummyDataGenerator(config)
    global_bc = generator.read_boundary_conditions()
    assert 0 in global_bc
    assert global_bc[0]["dirichlet"]["x"].shape == (3, 2)
    segment_bc = generator._process_segment_boundary(global_bc, 0)
    assert segment_bc[0]["dirichlet"]["x"].shape == (3, 2)
    assert np.allclose(segment_bc[0]["dirichlet"]["values"][:, 0], 1.0)


def test_segment_mask_includes_interface_points_on_both_adjacent_segments():
    config = DummyConfig()
    config.n_segments = [2, 1]
    config.x_min = [np.array([0.0, 0.0]), np.array([0.5, 0.0])]
    config.x_max = [np.array([0.5, 2.0]), np.array([1.0, 2.0])]
    generator = DummyDataGenerator(config)

    x = np.array([[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]])
    left_mask = generator._create_segment_mask(x, 0)
    right_mask = generator._create_segment_mask(x, 1)

    assert np.array_equal(left_mask, np.array([True, True, False]))
    assert np.array_equal(right_mask, np.array([False, True, True]))


def test_segment_local_boundary_sampling_uses_per_segment_face_points():
    config = DummyConfig()
    config.n_segments = [2, 2]
    config.x_min = [
        np.array([0.0, 0.0]),
        np.array([0.5, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.5, 1.0]),
    ]
    config.x_max = [
        np.array([0.5, 1.0]),
        np.array([1.0, 1.0]),
        np.array([0.5, 2.0]),
        np.array([1.0, 2.0]),
    ]
    config.boundary_conditions = [
        {
            "type": "dirichlet",
            "region": "left",
            "points": 3,
            "value": "x + 1",
            "sampling_mode": "segment_local",
        }
    ]
    generator = DummyDataGenerator(config)

    global_bc = generator.read_boundary_conditions()
    bottom_left = generator._process_segment_boundary(global_bc, 0)
    bottom_right = generator._process_segment_boundary(global_bc, 1)
    top_left = generator._process_segment_boundary(global_bc, 2)

    assert bottom_left[0]["dirichlet"]["x"].shape == (3, 2)
    assert top_left[0]["dirichlet"]["x"].shape == (3, 2)
    assert bottom_right[0]["dirichlet"]["x"].size == 0
    assert np.allclose(bottom_left[0]["dirichlet"]["x"][:, 0], 0.0)
    assert np.allclose(bottom_left[0]["dirichlet"]["x"][:, 1], [0.0, 0.5, 1.0])
    assert np.allclose(top_left[0]["dirichlet"]["x"][:, 0], 0.0)
    assert np.allclose(top_left[0]["dirichlet"]["x"][:, 1], [0.0, 0.5, 1.0])
    assert np.allclose(bottom_left[0]["dirichlet"]["values"][:, 0], 1.0)


def test_read_boundary_conditions_aggregates_segment_local_points_for_training():
    config = DummyConfig()
    config.n_segments = [2, 2]
    config.x_min = [
        np.array([0.0, 0.0]),
        np.array([0.5, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.5, 1.0]),
    ]
    config.x_max = [
        np.array([0.5, 1.0]),
        np.array([1.0, 1.0]),
        np.array([0.5, 2.0]),
        np.array([1.0, 2.0]),
    ]
    config.boundary_conditions = [
        {
            "type": "dirichlet",
            "region": "left",
            "points": 3,
            "value": "x + 1",
            "sampling_mode": "segment_local",
        }
    ]
    generator = DummyDataGenerator(config)

    global_bc = generator.read_boundary_conditions()

    assert global_bc[0]["dirichlet"]["x"].shape == (6, 2)
    assert np.allclose(global_bc[0]["dirichlet"]["x"][:, 0], 0.0)
    assert np.allclose(global_bc[0]["dirichlet"]["x"][:3, 1], [0.0, 0.5, 1.0])
    assert np.allclose(global_bc[0]["dirichlet"]["x"][3:, 1], [1.0, 1.5, 2.0])
    assert np.allclose(global_bc[0]["dirichlet"]["values"][:, 0], 1.0)
