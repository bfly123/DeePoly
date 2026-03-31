from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.utils.config import (
    FixIterationPDEConfig,
)


def build_fixiteration_config_stub():
    config = FixIterationPDEConfig.__new__(FixIterationPDEConfig)
    config.eq = {"L1": ["diff(u,x)"], "L2": ["0"], "F": ["1"], "N": ["u**2"]}
    config.vars_list = ["u", "v"]
    config.spatial_vars = ["x"]
    config.const_list = []
    config.max_iterations = 100
    config.residual_tol = 1e-6
    config.dt_init = 0.1
    config.dt_min = 1e-4
    config.dt_max = 0.5
    config.adaptive_dt = True
    config.dt_increase_factor = 1.2
    config.dt_decrease_factor = 0.5
    config.check_interval = 10
    config.print_interval = 20
    config.save_interval = 50
    config.max_residual = 1e4
    config.residual_stagnation_steps = 12
    config.time_scheme = "implicit_euler"
    config.initial_condition = {}
    config.fitter_training = {}
    config.epochs_adam = 1000
    config.epochs_lbfgs = 0
    config.DNNtol = 1e-4
    return config


def test_fixiteration_config_normalizes_operator_lists_and_aliases():
    config = build_fixiteration_config_stub()
    config._normalize_operator_splitting()
    assert config.eq_L1 == ["diff(u,x)", "diff(u,x)"]
    assert config.eq_L2 == ["0", "0"]
    assert config.eq_F == ["1", "1"]
    assert config.eq_N == ["u**2", "u**2"]
    assert config.f_L2 == config.eq_F
    assert config.N == config.eq_N


def test_fixiteration_config_loads_nested_fixiteration_block():
    config = build_fixiteration_config_stub()
    config.fixiteration = {
        "max_iterations": 250,
        "dt_init": 0.02,
        "dt_min": 0.001,
        "dt_max": 0.2,
        "time_scheme": "onestep_predictor",
    }
    config._load_pseudo_time_config()
    assert config.max_iterations == 250
    assert config.dt_init == 0.02
    assert config.dt_min == 0.001
    assert config.dt_max == 0.2
    assert config.time_scheme == "onestep_predictor"


def test_fixiteration_config_merges_fitter_training_overrides():
    config = build_fixiteration_config_stub()
    config.fitter_training = {"epochs": 50, "epochs_lbfgs": 5, "DNNtol": 1e-6}
    config._load_fitter_training_config()
    assert config.epochs_adam == 50
    assert config.epochs_lbfgs == 5
    assert config.DNNtol == 1e-6


def test_fixiteration_config_validate_rejects_invalid_dt_range():
    config = build_fixiteration_config_stub()
    config.dt_init = 1.0
    assert config.validate() is False
