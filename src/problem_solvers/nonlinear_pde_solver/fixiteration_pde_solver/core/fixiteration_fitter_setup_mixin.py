from typing import Dict

from src.algebraic_solver import LinearSolver


class FixIterationFitterSetupMixin:
    def _initialize_fixiteration_runtime(self, config, data: Dict = None) -> None:
        _ = data
        self.solver = LinearSolver(
            verbose=True,
            use_gpu=True,
            performance_tracking=True,
            config=config,
        )
        self.max_iterations = getattr(config, "max_iterations", 100)
        self.convergence_tol = getattr(config, "residual_tol", 1e-6)
        self.under_relaxation = float(getattr(config, "under_relaxation", 1.0))
        self.print_interval = getattr(config, "print_interval", 10)
        self.residual_history = []
