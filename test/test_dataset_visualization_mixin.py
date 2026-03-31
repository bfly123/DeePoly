import numpy as np

from src.abstract_class.config.visualize_mixins.dataset_visualization_mixin import (
    DatasetVisualizationMixin,
)


class _DummyVisConfig:
    colormap = "viridis"
    figure_dpi = 120
    plot_mode_train = "scatter"
    plot_mode_test = "grid"
    scatter_kwargs = {"s": 5, "alpha": 0.6}
    train_scatter_kwargs = {"alpha": 0.9}
    grid_kwargs = {"resolution": 50}
    test_grid_kwargs = {"resolution": 200}

    @staticmethod
    def should_plot_test():
        return True

    @staticmethod
    def should_plot_train():
        return True


class _TrainOnlyVisConfig(_DummyVisConfig):
    @staticmethod
    def should_plot_test():
        return False


class _RecordingPlotter:
    def __init__(self):
        self.calls = []

    def plot_solution(self, values, prepared, save_path, **kwargs):
        self.calls.append(
            {
                "values": values,
                "prepared": prepared,
                "save_path": save_path,
                "kwargs": kwargs,
            }
        )


class _Harness(DatasetVisualizationMixin):
    def __init__(self, vis_cfg):
        self.config = type("Config", (), {"visualization": vis_cfg})()
        self.dimensional_visualizer = _RecordingPlotter()

    @staticmethod
    def _to_numpy_cpu(values):
        return values


def test_prepare_plot_data_reconstructs_x_from_segments():
    harness = _Harness(_DummyVisConfig())
    prepared = harness._prepare_plot_data(
        {
            "x_segments": [
                np.array([[0.0], [0.5]]),
                np.array([[1.0]]),
            ]
        }
    )

    assert prepared is not None
    assert np.allclose(prepared["x"], np.array([[0.0], [0.5], [1.0]]))


def test_build_plot_kwargs_merges_global_and_dataset_specific_overrides():
    harness = _Harness(_DummyVisConfig())

    train_kwargs = harness._build_plot_kwargs("train")
    test_kwargs = harness._build_plot_kwargs("test")

    assert train_kwargs["plot_mode"] == "scatter"
    assert train_kwargs["scatter_kwargs"] == {"s": 5, "alpha": 0.9}
    assert train_kwargs["grid_kwargs"] == {"resolution": 50}
    assert test_kwargs["plot_mode"] == "grid"
    assert test_kwargs["grid_kwargs"] == {"resolution": 200}


def test_render_solution_outputs_records_train_and_test_plots(tmp_path):
    harness = _Harness(_DummyVisConfig())
    x = np.array([[0.0], [1.0]])
    predictions_test = {"u": np.array([[1.0], [2.0]])}
    predictions_train = {"u": np.array([[3.0], [4.0]])}

    render_info = harness.render_solution_outputs(
        outputs=["u"],
        predictions_test=predictions_test,
        predictions_train=predictions_train,
        data_test={"x": x},
        data_train={"x": x},
        results_dir=str(tmp_path),
    )

    assert "u" in render_info
    assert set(render_info["u"].keys()) == {"test", "train"}
    assert render_info["u"]["test"]["save_path"].endswith("u_solution.png")
    assert render_info["u"]["train"]["save_path"].endswith("u_train_solution.png")
    assert len(harness.dimensional_visualizer.calls) == 2


def test_render_solution_outputs_uses_shared_filename_when_only_train_is_enabled(tmp_path):
    harness = _Harness(_TrainOnlyVisConfig())
    x = np.array([[0.0], [1.0]])

    render_info = harness.render_solution_outputs(
        outputs=["u"],
        predictions_test={},
        predictions_train={"u": np.array([[3.0], [4.0]])},
        data_test={"x": x},
        data_train={"x": x},
        results_dir=str(tmp_path),
    )

    assert render_info["u"]["train"]["save_path"].endswith("u_solution.png")
