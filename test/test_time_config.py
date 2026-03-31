from dataclasses import fields

from src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils.config import (
    TimePDEConfig,
)
from src.problem_solvers.nonlinear_pde_solver.time_pde_solver.utils.visualize import (
    TimePDEVisualizer,
)


def build_time_config_stub():
    config = TimePDEConfig.__new__(TimePDEConfig)
    config.eq = {"L1": ["diff(u,x)"], "L2": ["0"], "F": ["1"], "N": ["u**2"]}
    config.vars_list = ["u", "v"]
    config.spatial_vars = ["x"]
    config.const_list = []
    config.T = 1.0
    config.dt = 0.1
    config.time_scheme = "IMEX_RK_2_2_2"
    config.time_scheme_params = {}
    config.epochs_adam = 100
    config.epochs_lbfgs = 10
    config.max_retries = 1
    config.spotter_skip = 2
    return config


def test_time_config_normalizes_operator_lists_and_aliases():
    config = build_time_config_stub()
    config._normalize_operator_splitting()
    assert config.eq_L1 == ["diff(u,x)", "diff(u,x)"]
    assert config.eq_L2 == ["0", "0"]
    assert config.eq_F == ["1", "1"]
    assert config.eq_N == ["u**2", "u**2"]
    assert config.f_L2 == config.eq_F
    assert config.N == config.eq_N


def test_time_config_sets_scheme_defaults_and_overrides():
    config = build_time_config_stub()
    config.time_scheme_params = {"max_time_steps": 25}
    config._setup_time_scheme_params()
    assert config.max_time_steps == 25
    assert config.time_output_interval == 10
    assert config.stability_check is True


def test_time_config_onestep_predictor_defaults():
    config = build_time_config_stub()
    config.time_scheme = "onestep_predictor"
    config._setup_time_scheme_params()
    assert config.max_time_steps == 11
    assert config.time_output_interval == 1
    assert config.stability_check is False


def test_time_pde_default_scheme_is_onestep_predictor():
    defaults = {field.name: field.default for field in fields(TimePDEConfig)}
    assert defaults["time_scheme"] == "onestep_predictor"


def test_time_visualizer_keeps_full_history_for_onestep_predictor():
    class DummyConfig:
        spatial_vars = ["x"]
        x_domain = [[-1.0, 1.0]]
        n_segments = [1]
        T = 1.0
        dt = 0.005
        time_scheme = "onestep_predictor"

    visualizer = TimePDEVisualizer(DummyConfig())

    assert visualizer._max_frames >= 201
