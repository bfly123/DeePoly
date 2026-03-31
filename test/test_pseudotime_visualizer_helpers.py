import matplotlib
import numpy as np
from matplotlib.collections import QuadMesh
import json
import pytest

matplotlib.use("Agg")

from src.abstract_class.config.visualization_config import VisualizationConfig
from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.pseudotime_solver_state_mixin import (
    PseudoTimeSolverStateMixin,
)
from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.utils.visualize import (
    PseudoTimePDEVisualizer,
)


class DummyVisualization:
    save_animation = True
    realtime_visualization = False
    animation_skip = 1
    colormap = "RdBu_r"
    reference_policy = "auto"
    outputs = ["u"]
    plot_mode_test = "grid"
    grid_kwargs = {"shading": "auto"}

    @staticmethod
    def should_plot_test():
        return True


class DummyConfig:
    def __init__(self):
        self.spatial_vars = ["x", "y"]
        self.n_segments = [1, 1]
        self.x_domain = [[0.0, 1.0], [0.0, 1.0]]
        self.visualization = DummyVisualization()
        self.results_dir = "."
        self.vars_list = ["u"]
        self.n_eqs = 1
        self.constants = {"Re": 100.0}


def test_visualization_config_defaults_to_realtime_enabled():
    assert VisualizationConfig().realtime_visualization is True


def test_pseudotime_reference_interpolation_to_grid():
    visualizer = PseudoTimePDEVisualizer(DummyConfig())
    reference = {
        "x_ref": np.array([0.0, 1.0]),
        "y_ref": np.array([0.0, 1.0]),
        "u_ref": np.array([[1.0, 2.0], [3.0, 4.0]]),
    }
    x_unique = np.array([0.0, 1.0])
    y_unique = np.array([0.0, 1.0])
    _, _, ref_grid = visualizer._interpolate_reference_to_grid(reference, x_unique, y_unique)
    assert ref_grid.shape == (2, 2)
    assert np.allclose(ref_grid, np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_pseudotime_reference_evaluation_at_coords():
    visualizer = PseudoTimePDEVisualizer(DummyConfig())
    coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    reference = {
        "x_ref": np.array([0.0, 1.0]),
        "y_ref": np.array([0.0, 1.0]),
        "u_ref": np.array([[1.0, 2.0], [3.0, 4.0]]),
    }
    values = visualizer._evaluate_reference_at_coords(coords, reference)
    assert values.shape == (2, 1)
    assert np.allclose(values[:, 0], [1.0, 4.0])


def test_pseudotime_animation_frame_uses_grid_rendering_for_structured_mesh():
    visualizer = PseudoTimePDEVisualizer(DummyConfig())
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    snapshot = np.array([[1.0], [2.0], [3.0], [4.0]])
    visualizer._anim_coords = coords
    visualizer._anim_iterations = [0]
    visualizer._anim_residuals = [1e-2]

    fig = visualizer._render_animation_frame(coords, snapshot, 0)

    assert any(isinstance(collection, QuadMesh) for collection in fig.axes[0].collections)
    visualizer._close_figure(fig)


def test_pseudotime_realtime_2d_display_expands_to_full_domain():
    config = DummyConfig()
    config.visualization.realtime_visualization = True
    visualizer = PseudoTimePDEVisualizer(config)
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    snapshot = np.array([[1.0], [2.0], [3.0], [4.0]])

    visualizer.update_animation_data(0, 1e-2, snapshot, {"x": coords}, 1, None)

    ax = visualizer._realtime_axes["prediction"]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert xlim[0] < 0.0 and xlim[1] > 1.0
    assert ylim[0] < 0.0 and ylim[1] > 1.0
    visualizer._close_figure(visualizer._realtime_fig)


def test_pseudotime_flow_diagnostics_generate_streamline_and_centerline_plots(tmp_path):
    config = DummyConfig()
    config.results_dir = str(tmp_path)
    config.visualization.outputs = ["sniper"]
    config.vars_list = ["u", "v", "p"]
    config.n_eqs = 3
    visualizer = PseudoTimePDEVisualizer(config)

    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 1.0, 9)
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])
    psi = np.sin(np.pi * coords[:, 0]) * np.sin(np.pi * coords[:, 1])
    prediction = np.column_stack(
        [
            np.pi * np.sin(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1]),
            -np.pi * np.cos(np.pi * coords[:, 0]) * np.sin(np.pi * coords[:, 1]),
            psi,
        ]
    )

    visualizer._save_flow_diagnostics(
        "sniper",
        {"x": coords},
        prediction,
        str(tmp_path),
    )

    assert (tmp_path / "sniper_flow_summary.png").is_file()
    assert (tmp_path / "sniper_centerline_profiles.png").is_file()
    assert (tmp_path / "sniper_flow_field_structured.npz").is_file()
    assert (tmp_path / "sniper_flow_field.dat").is_file()
    assert (tmp_path / "sniper_streamfunction_global.png").is_file()
    assert (tmp_path / "sniper_corner_vortices.png").is_file()
    assert (tmp_path / "sniper_vortex_benchmark.json").is_file()
    assert (tmp_path / "sniper_vortex_comparison.png").is_file()
    assert (tmp_path / "sniper_vortex_comparison.pdf").is_file()

    structured = np.load(tmp_path / "sniper_flow_field_structured.npz")
    assert structured["x_mesh"].shape == (9, 9)
    assert structured["y_mesh"].shape == (9, 9)
    assert structured["u"].shape == (9, 9)
    assert structured["v"].shape == (9, 9)
    assert structured["p"].shape == (9, 9)
    assert structured["speed"].shape == (9, 9)
    assert structured["vorticity"].shape == (9, 9)
    assert structured["streamfunction"].shape == (9, 9)
    assert structured["component_names"].tolist() == ["u", "v", "p"]
    np.testing.assert_allclose(structured["streamfunction"], psi.reshape(9, 9), atol=6.0e-2, rtol=6.0e-2)

    tecplot_text = (tmp_path / "sniper_flow_field.dat").read_text(encoding="utf-8")
    assert 'VARIABLES = "X", "Y", "u", "v", "p", "speed", "vorticity", "streamfunction"' in tecplot_text
    assert "ZONE I=9, J=9, F=POINT" in tecplot_text


def test_re1000_vortex_benchmark_matches_ghia_table_v():
    config = DummyConfig()
    config.constants = {"Re": 1000.0}
    visualizer = PseudoTimePDEVisualizer(config)

    benchmark = visualizer._streamfunction_benchmark_data()

    assert visualizer._streamfunction_benchmark_source() == "Ghia, Ghia, and Shin (1982) Table V"
    assert benchmark is not None
    assert benchmark["primary"]["x"] == pytest.approx(0.5313)
    assert benchmark["primary"]["y"] == pytest.approx(0.5625)
    assert benchmark["primary"]["psi"] == pytest.approx(-1.17929e-1)
    assert benchmark["bottom_left_secondary"]["x"] == pytest.approx(0.0859)
    assert benchmark["bottom_left_secondary"]["y"] == pytest.approx(0.0781)
    assert benchmark["bottom_right_secondary"]["x"] == pytest.approx(0.8594)
    assert benchmark["bottom_right_secondary"]["y"] == pytest.approx(0.1094)
    assert "bottom_left_tertiary" not in benchmark
    assert benchmark["bottom_right_tertiary"]["x"] == pytest.approx(0.9922)
    assert benchmark["bottom_right_tertiary"]["y"] == pytest.approx(0.0078)


def test_re400_vortex_benchmark_matches_ghia_table_v():
    config = DummyConfig()
    config.constants = {"Re": 400.0}
    visualizer = PseudoTimePDEVisualizer(config)

    benchmark = visualizer._streamfunction_benchmark_data()

    assert visualizer._streamfunction_benchmark_source() == "Ghia, Ghia, and Shin (1982) Table V"
    assert benchmark is not None
    assert benchmark["primary"]["x"] == pytest.approx(0.5547)
    assert benchmark["primary"]["y"] == pytest.approx(0.6055)
    assert benchmark["primary"]["psi"] == pytest.approx(-1.13909e-1)
    assert benchmark["bottom_left_secondary"]["x"] == pytest.approx(0.0508)
    assert benchmark["bottom_left_secondary"]["y"] == pytest.approx(0.0469)
    assert benchmark["bottom_left_secondary"]["psi"] == pytest.approx(1.41951e-5)
    assert benchmark["bottom_right_secondary"]["x"] == pytest.approx(0.8906)
    assert benchmark["bottom_right_secondary"]["y"] == pytest.approx(0.1250)
    assert benchmark["bottom_right_secondary"]["psi"] == pytest.approx(6.42352e-4)


class _RecordingVisualizer:
    def __init__(self):
        self.calls = []

    def update_animation_data(self, iteration, residual, U, data, skip, reference_solution=None):
        self.calls.append(
            {
                "iteration": iteration,
                "residual": residual,
                "U": U,
                "data": data,
                "skip": skip,
                "reference_solution": reference_solution,
            }
        )


class _AnimationHarness(PseudoTimeSolverStateMixin):
    def __init__(self):
        self.config = type("Cfg", (), {"visualization": DummyVisualization()})()
        self.visualizer = _RecordingVisualizer()
        self.data_train = {"x": np.array([[0.0, 0.0], [1.0, 1.0]])}
        self.data_test = {
            "x": np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ]
            )
        }
        self.reference_solution = None

    @staticmethod
    def _to_numpy_cpu(data):
        return np.asarray(data)

    def _interpolate_to_test_grid(self, U):
        mean_value = float(np.mean(U))
        return np.full((self.data_test["x"].shape[0], 1), mean_value)


def test_pseudotime_solver_animation_prefers_test_mesh_payload():
    harness = _AnimationHarness()
    U_train = np.array([[1.0], [3.0]])

    harness._update_animation_frame(iteration=2, residual=1e-3, U=U_train, animation_skip=2)

    assert len(harness.visualizer.calls) == 1
    call = harness.visualizer.calls[0]
    assert call["data"] is harness.data_test
    assert call["U"].shape == (4, 1)
    assert np.allclose(call["U"], 2.0)


def test_pseudotime_save_data_writes_convergence_info_json(tmp_path):
    config = DummyConfig()
    config.results_dir = str(tmp_path)
    visualizer = PseudoTimePDEVisualizer(config)
    results = {
        "coefficients": np.array([[1.0]]),
        "convergence_info": {
            "converged": False,
            "final_tau": 1.2,
            "tau_history": [0.0, 0.1],
            "residual_history": [1.0, 0.5],
        },
        "predictions": {"sniper": np.array([[2.0], [3.0]])},
        "predictions_train": {"sniper": np.array([[4.0], [5.0]])},
    }
    data = {"x": np.array([[0.0, 0.0], [1.0, 1.0]])}

    visualizer._save_data(results, data, data, str(tmp_path))

    info = json.loads((tmp_path / "convergence_info.json").read_text(encoding="utf-8"))
    assert info["converged"] is False
    assert info["final_tau"] == 1.2
    assert info["tau_history"] == [0.0, 0.1]
