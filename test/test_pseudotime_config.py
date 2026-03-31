from types import SimpleNamespace

from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.solver import (
    PseudoTimePDESolver,
)
from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.utils.config import (
    PseudoTimePDEConfig,
)


def build_pseudotime_config_stub():
    config = PseudoTimePDEConfig.__new__(PseudoTimePDEConfig)
    config.eq = {"L1": ["diff(u,x)"], "L2": ["0"], "F": ["1"], "N": ["u**3"]}
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
    config.convergence_primary_var = 0
    config.convergence_primary_tol = 1e-4
    config.convergence_improvement_factor = 0.999
    config.convergence_plateau_steps = 10
    config.convergence_plateau_tol = 1e-2
    config.convergence_dt_growth_tol = 1e-3
    config.rollback_on_plateau = True
    config.rollback_dt_min = 1e-2
    config.force_retrain_on_plateau = True
    config.return_best_on_exit = True
    config.reference_stable_single_retrain = True
    config.reference_stable_retrain_stop_tol = 0.0
    config.reference_stable_adaptive_tail = False
    config.time_scheme = "implicit_euler"
    config.steady_state_mode = False
    config.pseudo_time_formulation = "default"
    config.mass_eq_mask = []
    config.mass_var_map = []
    config.frozen_nonlinear_jacobian = "none"
    config.initial_condition = {}
    config.spotter = {}
    config.fitter_training = {}
    return config


def test_pseudotime_config_normalizes_operator_lists_and_aliases():
    config = build_pseudotime_config_stub()

    config._normalize_operator_splitting()

    assert config.eq_L1 == ["diff(u,x)", "diff(u,x)"]
    assert config.eq_L2 == ["0", "0"]
    assert config.eq_F == ["1", "1"]
    assert config.eq_N == ["u**3", "u**3"]
    assert config.f_L2 == config.eq_F
    assert config.N == config.eq_N


def test_pseudotime_config_loads_nested_pseudo_time_block():
    config = build_pseudotime_config_stub()
    config.pseudo_time = {
        "max_iterations": 250,
        "dt_init": 0.02,
        "dt_min": 0.001,
        "dt_max": 0.2,
        "steady_state_mode": True,
        "time_scheme": "onestep_predictor",
        "pseudo_time_formulation": "reference_stable",
        "mass_eq_mask": [1],
        "mass_var_map": [-1, 0],
        "frozen_nonlinear_jacobian": "transport",
        "convergence_primary_var": 1,
        "convergence_primary_tol": 5e-5,
        "convergence_plateau_steps": 7,
        "rollback_dt_min": 0.01,
        "reference_stable_adaptive_tail": True,
    }

    config._load_pseudo_time_config()

    assert config.max_iterations == 250
    assert config.dt_init == 0.02
    assert config.dt_min == 0.001
    assert config.dt_max == 0.2
    assert config.steady_state_mode is True
    assert config.time_scheme == "implicit_euler"
    assert config.pseudo_time_formulation == "reference_stable"
    assert config.mass_eq_mask == [1]
    assert config.mass_var_map == [-1, 0]
    assert config.frozen_nonlinear_jacobian == "transport"
    assert config.convergence_primary_var == 1
    assert config.convergence_primary_tol == 5e-5
    assert config.convergence_plateau_steps == 7
    assert config.rollback_dt_min == 0.01
    assert config.reference_stable_adaptive_tail is True


def test_pseudotime_config_reference_stable_uses_old_project_control_defaults():
    config = build_pseudotime_config_stub()
    config.dt_increase_factor = 1.2
    config.force_retrain_on_plateau = True
    config.pseudo_time = {
        "steady_state_mode": True,
        "pseudo_time_formulation": "reference_stable",
    }

    config._load_pseudo_time_config()

    assert config.dt_increase_factor == 2.0
    assert config.force_retrain_on_plateau is False
    assert config.rollback_dt_min == 0.01


def test_pseudotime_config_prefers_spotter_block_for_training_schedule():
    config = build_pseudotime_config_stub()
    config.spotter = {
        "train_every": 4,
        "epochs_adam": 80,
        "epochs_lbfgs": 6,
        "max_retries": 2,
        "tol": 5e-6,
    }

    config._load_spotter_config()

    assert config.spotter_skip == 4
    assert config.epochs_adam == 80
    assert config.epochs_lbfgs == 6
    assert config.max_retries == 2
    assert config.DNNtol == 5e-6
    assert config.spotter == {
        "train_every": 4,
        "epochs_adam": 80,
        "epochs_lbfgs": 6,
        "max_retries": 2,
        "tol": 5e-6,
    }


def test_pseudotime_config_keeps_legacy_fitter_training_compatibility():
    config = build_pseudotime_config_stub()
    config.fitter_training = {"epochs": 50, "epochs_lbfgs": 5, "DNNtol": 1e-6, "max_retries": 3}

    config._load_fitter_training_config()

    assert config.spotter_skip == 1
    assert config.epochs_adam == 50
    assert config.epochs_lbfgs == 5
    assert config.max_retries == 3
    assert config.DNNtol == 1e-6
    assert config.spotter == {
        "train_every": 1,
        "epochs_adam": 50,
        "epochs_lbfgs": 5,
        "max_retries": 3,
        "tol": 1e-6,
    }


def test_pseudotime_config_validate_rejects_invalid_dt_range():
    config = build_pseudotime_config_stub()
    config.dt_init = 1.0

    assert config.validate() is False


def test_pseudotime_solver_spotter_schedule_uses_periodic_skip_from_first_iteration():
    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.config = SimpleNamespace(spotter_skip=5)

    schedule = [solver._should_train_network(iteration) for iteration in range(12)]

    assert schedule == [
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
    ]


def test_pseudotime_solver_reference_stable_schedule_matches_legacy_retrain_pattern():
    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.config = SimpleNamespace(
        spotter_skip=10,
        steady_state_mode=True,
        pseudo_time_formulation="reference_stable",
        convergence_plateau_tol=1e-2,
    )
    solver._reference_stable_last_primary_update = float("inf")
    solver._reference_stable_post_threshold_retrain_done = False

    assert solver._should_train_network(0) is True
    assert solver._should_train_network(1) is False

    solver._reference_stable_last_primary_update = 5e-3
    assert solver._should_train_network(7) is True

    solver._on_network_trained(7, 0.02, None, 1e-3)
    assert solver._should_train_network(8) is False


def test_reference_stable_rejected_retrain_uses_cooldown_and_retries():
    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.config = SimpleNamespace(
        spotter_skip=5,
        steady_state_mode=True,
        pseudo_time_formulation="reference_stable",
        convergence_plateau_tol=1e-2,
        DNNtol=0.03,
    )
    solver._reference_stable_last_primary_update = 5e-3
    solver._reference_stable_post_threshold_retrain_done = False
    solver._reference_stable_last_retrain_iter = None

    assert solver._should_train_network(6) is True

    solver._on_network_trained(6, 0.02, None, 3.5e-2, accepted=False)

    assert solver._reference_stable_post_threshold_retrain_done is False
    assert solver._should_train_network(7) is False
    assert solver._should_train_network(10) is False
    assert solver._should_train_network(11) is True


def test_reference_stable_accepted_retrain_finishes_schedule():
    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.config = SimpleNamespace(
        spotter_skip=5,
        steady_state_mode=True,
        pseudo_time_formulation="reference_stable",
        convergence_plateau_tol=1e-2,
        DNNtol=0.04,
    )
    solver._reference_stable_last_primary_update = 5e-3
    solver._reference_stable_post_threshold_retrain_done = False
    solver._reference_stable_last_retrain_iter = None

    assert solver._should_train_network(6) is True

    solver._on_network_trained(6, 0.02, None, 3.0e-2, accepted=True)

    assert solver._reference_stable_post_threshold_retrain_done is True
    assert solver._should_train_network(11) is False


def test_reference_stable_repeat_mode_keeps_tail_retrain_schedule_active():
    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.config = SimpleNamespace(
        spotter_skip=5,
        steady_state_mode=True,
        pseudo_time_formulation="reference_stable",
        convergence_plateau_tol=1e-2,
        DNNtol=0.04,
        reference_stable_single_retrain=False,
    )
    solver._reference_stable_last_primary_update = 5e-3
    solver._reference_stable_post_threshold_retrain_done = False
    solver._reference_stable_last_retrain_iter = None

    assert solver._should_train_network(6) is True

    solver._on_network_trained(6, 0.02, None, 3.0e-2, accepted=True)

    assert solver._reference_stable_post_threshold_retrain_done is False
    assert solver._should_train_network(7) is False
    assert solver._should_train_network(11) is True


def test_reference_stable_skip_retrain_when_primary_update_is_already_small():
    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.config = SimpleNamespace(
        spotter_skip=5,
        steady_state_mode=True,
        pseudo_time_formulation="reference_stable",
        convergence_plateau_tol=1e-2,
        reference_stable_retrain_stop_tol=1e-3,
    )
    solver._reference_stable_last_primary_update = 5e-4
    solver._reference_stable_post_threshold_retrain_done = False
    solver._reference_stable_last_retrain_iter = None

    assert solver._should_train_network(11) is False


def test_reference_stable_transfer_can_skip_initial_training():
    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.config = SimpleNamespace(
        spotter_skip=5,
        steady_state_mode=True,
        pseudo_time_formulation="reference_stable",
        convergence_plateau_tol=1e-2,
    )
    solver._skip_initial_spotter_training = True

    assert solver._should_train_network(0) is False


def test_reference_stable_retrain_update_is_rejected_when_loss_exceeds_dnntol():
    solver = PseudoTimePDESolver.__new__(PseudoTimePDESolver)
    solver.config = SimpleNamespace(
        steady_state_mode=True,
        pseudo_time_formulation="reference_stable",
        DNNtol=0.03,
    )

    assert solver._accept_trained_model_update(0, 0.03, None, 0.05) is True
    assert solver._accept_trained_model_update(7, 0.03, None, 0.05) is False
    assert solver._accept_trained_model_update(7, 0.03, None, 0.02) is True
