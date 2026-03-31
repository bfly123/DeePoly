import numpy as np
import torch

from src.abstract_class.config.base_data_boundary_mixin import BaseDataBoundaryMixin
from src.abstract_class.config.base_data_segment_mixin import BaseDataSegmentMixin
from src.abstract_class.operator_factory import create_operator_factory
from src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils.data import TimePDEDataGenerator
from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.core.net import (
    PseudoTimePDENet,
)
from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.core.pseudotime_fitter_jacobian_mixin import (
    PseudoTimeFitterJacobianMixin,
)


class _BoundaryConfig:
    vars_list = ["u", "v", "p"]
    spatial_vars = ["x", "y"]
    x_domain = [[0.0, 1.0], [0.0, 1.0]]
    n_dim = 2
    n_segments = [1, 1]
    boundary_conditions = [
        {"type": "dirichlet", "region": "left", "var": "u", "value": 1.0, "points": 3},
        {"type": "dirichlet", "region": "right", "vars": ["v", "p"], "value": 0.0, "points": 2},
    ]


class _BoundaryHarness(BaseDataBoundaryMixin):
    def __init__(self):
        self.config = _BoundaryConfig()
        self.n_dim = self.config.n_dim
        self.n_segments = self.config.n_segments

    def _parse_math_expression(self, expr, x_boundary):
        value = float(expr)
        return np.full((x_boundary.shape[0], 1), value)


def test_boundary_reader_respects_var_and_vars_fields():
    harness = _BoundaryHarness()
    boundary = harness.read_boundary_conditions()

    assert boundary[0]["dirichlet"]["x"].shape[0] == 3
    assert boundary[1]["dirichlet"]["x"].shape[0] == 2
    assert boundary[2]["dirichlet"]["x"].shape[0] == 2


class _SegmentBoundaryConfig:
    vars_list = ["u"]
    spatial_vars = ["x", "y"]
    x_domain = [[0.0, 1.0], [0.0, 1.0]]
    x_min = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
        ]
    )
    x_max = np.array(
        [
            [0.5, 0.5],
            [1.0, 0.5],
            [0.5, 1.0],
            [1.0, 1.0],
        ]
    )
    n_dim = 2
    n_segments = [2, 2]
    boundary_conditions = [
        {
            "type": "dirichlet",
            "region": "left",
            "var": "u",
            "value": 1.0,
            "points": 3,
            "sampling_mode": "segment_local",
        },
    ]


class _SegmentBoundaryHarness(BaseDataBoundaryMixin, BaseDataSegmentMixin):
    def __init__(self):
        self.config = _SegmentBoundaryConfig()
        self.n_dim = self.config.n_dim
        self.n_segments = self.config.n_segments
        self.Ns = int(np.prod(self.n_segments))

    def _parse_math_expression(self, expr, x_boundary):
        value = float(expr)
        return np.full((x_boundary.shape[0], 1), value)


def test_segment_boundary_helpers_map_named_and_dim_regions_to_segments():
    harness = _SegmentBoundaryHarness()

    assert harness._region_to_segment("left") == 0
    assert harness._region_to_segment("right") == 1
    assert harness._region_to_segment("top") == 2
    assert harness._region_to_segment("dim0_max") == 1
    assert harness._region_to_segment("dim1_max") == 2


def test_segment_boundary_local_sampling_only_generates_domain_edge_points():
    harness = _SegmentBoundaryHarness()

    left_edge = harness._generate_segment_boundary_points("left", 3, 0)
    interior_edge = harness._generate_segment_boundary_points("left", 3, 1)
    segment_boundary = harness._process_segment_boundary_locally({}, 0)

    assert left_edge.shape == (3, 2)
    assert interior_edge.size == 0
    assert np.allclose(left_edge[:, 0], 0.0)
    assert segment_boundary[0]["dirichlet"]["x"].shape == (3, 2)
    assert np.allclose(segment_boundary[0]["dirichlet"]["x"][:, 0], 0.0)


def test_segment_boundary_projection_normalizes_global_points_per_segment():
    harness = _SegmentBoundaryHarness()
    harness.config.boundary_conditions = []
    global_boundary = {0: harness._empty_boundary_entry()}
    global_boundary[0]["dirichlet"]["x"] = np.array([[0.0, 0.0], [0.0, 0.25], [0.0, 0.75]])
    global_boundary[0]["dirichlet"]["values"] = np.ones((3, 1))

    projected = harness._process_segment_boundary(global_boundary, 0)

    assert projected[0]["dirichlet"]["x"].shape == (2, 2)
    assert np.allclose(projected[0]["dirichlet"]["x"][:, 0], 0.0)
    assert np.allclose(projected[0]["dirichlet"]["x"][:, 1], [0.0, 0.5])
    assert projected[0]["dirichlet"]["values"].shape == (2, 1)


class _TimeConfig:
    def __init__(self):
        self.vars_list = ["u", "v", "p"]
        self.n_eqs = 3
        self.spatial_vars = ["x", "y"]
        self.x_domain = [[0.0, 1.0], [0.0, 1.0]]
        self.Initial_conditions = [
            {"var": "u", "value": "x"},
            {"var": "v", "value": "y"},
            {"var": "p", "value": "x + y"},
        ]
        self.n_dim = 2


def test_time_data_generator_supports_multi_variable_initial_conditions():
    generator = TimePDEDataGenerator.__new__(TimePDEDataGenerator)
    generator.config = _TimeConfig()
    generator.n_eqs = 3
    x_global = np.array([[0.0, 0.5], [1.0, 0.25]])

    field = generator.generate_global_field(x_global)

    assert field.shape == (2, 3)
    assert np.allclose(field[:, 0], [0.0, 1.0])
    assert np.allclose(field[:, 1], [0.5, 0.25])
    assert np.allclose(field[:, 2], [0.5, 1.25])


class _PseudoConfig:
    n_eqs = 2
    n_dim = 1
    hidden_dims = [4]
    device = "cpu"
    vars_list = ["u", "v"]
    problem_type = "general_pde"
    steady_state_mode = True


class _PseudoNet(PseudoTimePDENet):
    def compute_pde_operators(self, U, x_train):
        del x_train
        return {
            "L1": [U[:, 0] + 1.0, U[:, 1] - 2.0],
            "L2": [0.0, 0.0],
            "F": [0.0, 0.0],
            "N": [0.0, 0.0],
            "S": [0.0, 0.0],
        }

    def _compute_boundary_loss(self, data_GPU):
        del data_GPU
        return torch.tensor(0.0, dtype=torch.float64)


def test_pseudotime_physics_loss_uses_steady_state_residual_without_u_current():
    config = _PseudoConfig()
    model = _PseudoNet(config)
    x_train = torch.tensor([[0.0], [1.0]], dtype=torch.float64, requires_grad=True)
    data_gpu = {
        "x_train": x_train,
        "U_current": torch.tensor([[10.0, 10.0], [10.0, 10.0]], dtype=torch.float64),
    }

    loss = model.physics_loss(data_gpu, dt=0.5)

    _, prediction = model(x_train)
    expected = torch.mean((prediction[:, 0] + 1.0) ** 2) + torch.mean((prediction[:, 1] - 2.0) ** 2)
    assert torch.allclose(loss, expected)


class _JacobianHarness(PseudoTimeFitterJacobianMixin):
    def __init__(self):
        self.config = type("Config", (), {"n_eqs": 1, "steady_state_mode": True})()
        self.dgN = 2
        self.data = {"x_segments_norm": [np.zeros((3, 1))]}
        self._linear_operators = {0: {"L1": [np.zeros((3, 2))], "L2": [np.zeros((3, 2))]}}
        self._features = {
            0: [
                np.ones((3, 2)),
                np.full((3, 2), 2.0),
            ]
        }
        self.received_features = None

    def F_func(self, features, u_seg):
        del u_seg
        self.received_features = features
        return np.zeros((3, 1))

    def N_func(self, features, u_seg):
        del u_seg
        self.received_features = features
        return [np.zeros(3)]

    def _get_source_term(self, segment_idx):
        del segment_idx
        return np.zeros((3, 1))


def test_pseudotime_segment_jacobian_passes_feature_list_to_nonlinear_operators():
    harness = _JacobianHarness()
    u_seg = [np.zeros((3, 1))]

    jacobian, rhs = harness._build_segment_jacobian(0, U_n_seg=u_seg, dt=0.1)

    assert harness.received_features is harness._features[0]
    assert jacobian.shape == (3, 2)
    assert rhs.shape == (3,)


def test_operator_factory_places_cross_variable_linear_terms_in_correct_blocks():
    all_derivatives = {
        "U_x": [0, 1],
        "V_y": [1, 2],
    }
    operator_terms = {
        "L1": [
            {
                "derivative_indices": [1, 2],
                "symbolic_expr": "U_x + V_y",
            }
        ]
    }
    features = [
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.full((2, 3), 2.0),
    ]
    factory = create_operator_factory(all_derivatives, optimized=True)

    linear_op = factory.create_all_operators(operator_terms)["L1_func"]
    matrix = linear_op(features)

    assert matrix.shape == (1, 2, 6)
    assert np.allclose(matrix[0, :, :3], np.ones((2, 3)))
    assert np.allclose(matrix[0, :, 3:], np.full((2, 3), 2.0))
