from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.algebraic_solver import LinearSolver


class TimePDEFitter(BaseDeepPolyFitter):
    """Time-dependent problem hybrid fitter implementation with pluggable time schemes

    Simplified version - time_step logic now in BaseDeepPolyFitter.
    This class only adds time-PDE-specific initialization and utilities.
    """

    def __init__(self, config, data: Dict = None, time_scheme: str = "onestep_predictor"):
        # Initialize base class with time_scheme support
        super().__init__(config, dt=None, data=data, time_scheme=time_scheme)

        # Initialize solver (specific to time-PDE solver)
        # 默认关闭verbose以减少重复输出，保留performance_tracking用于统计
        verbose_linear = getattr(config, 'verbose_linear_solver', False)
        self.solver = LinearSolver(
            verbose=verbose_linear,
            use_gpu=True,
            performance_tracking=True,
            config=config,
        )

    def solve_time_step(self, U_n: np.ndarray, U_seg: List[np.ndarray], dt: float,
                       coeffs_n: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Solve one time step - DEPRECATED wrapper

        .. deprecated::
            Use `time_step()` directly instead. This wrapper adds no value and will be
            removed in a future version.

        Delegates to base class time_step() method with precompilation check.
        """
        import warnings
        warnings.warn(
            "solve_time_step() is deprecated and will be removed in a future version. "
            "Use time_step() directly instead.",
            DeprecationWarning,
            stacklevel=2
        )

        if not self._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")

        return self.time_step(U_n, U_seg, dt, coeffs_n, **kwargs)

    # ================== Time Scheme Info Methods - DEPRECATED ==================
    # These methods have been moved to the solver layer for better separation of concerns.
    # They are kept here temporarily for backward compatibility but will be removed.

    def get_time_scheme_info(self) -> Dict[str, Any]:
        """Get time integration scheme information - DEPRECATED

        .. deprecated::
            Use `solver.get_solver_time_info()` instead. Time scheme information
            should be accessed at the solver layer, not the fitter layer.
        """
        import warnings
        warnings.warn(
            "get_time_scheme_info() is deprecated. Use solver.get_solver_time_info() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        if self.time_scheme is None:
            return {"error": "No time scheme initialized"}
        return self.time_scheme.get_scheme_info()

    def validate_operators(self) -> Dict[str, bool]:
        """Validate operator configuration - DEPRECATED

        .. deprecated::
            Use `solver.validate_operator_configuration()` instead. Validation logic
            belongs at the solver layer.
        """
        import warnings
        warnings.warn(
            "validate_operators() is deprecated. Use solver.validate_operator_configuration() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        if self.time_scheme is None:
            return {"error": "No time scheme initialized"}
        return self.time_scheme.validate_operators()

    def estimate_stable_dt(self, U_current: np.ndarray, safety_factor: float = 0.8) -> float:
        """Estimate stable time step size - DEPRECATED

        .. deprecated::
            Use `solver.estimate_stable_dt()` instead. Time step control belongs
            at the solver layer.
        """
        import warnings
        warnings.warn(
            "estimate_stable_dt() is deprecated. Use solver.estimate_stable_dt() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        if self.time_scheme is None:
            raise RuntimeError("No time scheme initialized")
        return self.time_scheme.estimate_stable_dt(U_current, safety_factor)

    def print_time_scheme_summary(self):
        """Print time integration scheme summary - DEPRECATED

        .. deprecated::
            Use `solver.print_time_scheme_summary()` instead. UI/reporting logic
            belongs at the solver layer.
        """
        import warnings
        warnings.warn(
            "print_time_scheme_summary() is deprecated. Use solver.print_time_scheme_summary() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        if self.time_scheme is None:
            print("No time scheme initialized")
            return
        self.time_scheme.print_scheme_summary()

    def _build_segment_jacobian(
        self, segment_idx: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build segment Jacobian matrix using the time scheme

        Delegates to the configured time integration scheme for time-specific handling.
        This implements the abstract method from BaseDeepPolyFitter.
        """
        if self.time_scheme is None:
            raise RuntimeError("Time scheme not initialized")
        return self.time_scheme.build_stage_jacobian(segment_idx, **kwargs)
