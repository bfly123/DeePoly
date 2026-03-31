"""
One-Step Predictor-Corrector Time Integration Scheme

One-step predictor-corrector time integration scheme:
- Predictor step: Implicit solve (u^* - u^n)/Δt + L1(u^*) + L2(u^*)⋅F(u^n) + N(u^n) = 0
- Corrector step: Explicit compute u^{n+1} = u^n - Δt * [L1_avg + L2F_avg + N_mid]
- Operator value pre-computation and time step reuse, reducing 75% operator calculations
"""

from typing import List, Tuple
import numpy as np

from .base_time_scheme import BaseTimeScheme
from .onestep_predictor_corrector_mixin import OneStepPredictorCorrectorMixin
from .onestep_predictor_operator_mixin import OneStepPredictorOperatorMixin
from .onestep_predictor_support_mixin import OneStepPredictorSupportMixin


class OneStepPredictor(
    OneStepPredictorSupportMixin,
    OneStepPredictorCorrectorMixin,
    OneStepPredictorOperatorMixin,
    BaseTimeScheme,
):
    """One-step predictor-corrector time integration scheme

    Features:
    - Predictor step implicit solve, corrector step explicit computation
    - Conditional correction: execute corrector step only when step > 1
    - Operator value pre-computation and time step reuse
    - Midpoint evaluation improves nonlinear term accuracy
    """

    def __init__(self, config):
        super().__init__(config)
        self.order = 1  # Basic accuracy order
        self.predictor_solution = None  # Store predictor solution u^*
        self.predictor_coeffs = None    # Store predictor coefficients β^*
        self.L1_beta_prev = None        # Store previous time step L1*β^n
        self.L2_beta_prev = None        # Store previous time step L2*β^n
        self.L1_beta_star = None        # Store current predictor step L1*β^*
        self.L2_beta_star = None        # Store current predictor step L2*β^*

    def time_step(
        self,
        U_n: np.ndarray,
        U_seg: List[np.ndarray],
        dt: float,
        coeffs_n: np.ndarray = None,
        current_time: float = 0.0,
        step: int = 0
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Execute one-step predictor-corrector time integration

        Args:
            U_n: Current time step global solution values
            U_seg: Current time step segment-level solution list
            dt: Time step size
            coeffs_n: Current time step coefficients (optional)
            current_time: Current time
            step: Time step number

        Returns:
            Tuple: (new global solution, new segment-level solution, new coefficients)
        """

        # StoreCurrent time和Solutionvalue（dtreduction handled in solver）
        self._current_time = current_time
        self.U_seg_n = [seg.copy() for seg in U_seg]

        # Step1: Predictor step（implicit solve）
        U_star_seg, coeffs_star = self._predictor_step(U_seg, dt, step)

        # Step2: Corrector step（only whenstep > 1execute explicit computation）
        if step > 1 and self.L1_beta_prev is not None:
            U_new_seg, coeffs_new = self._corrector_step(
                U_seg, U_star_seg, dt, coeffs_star, step
            )
        else:
            # First step directly uses predictor values
            U_new_seg, coeffs_new = U_star_seg, coeffs_star

        # Step3: Update operator values for next time step use
        # Current time步的Predictionoperator values成为Down一Time step的"previous"operator values
        if hasattr(self, 'L1_beta_star') and self.L1_beta_star is not None:
            self.L1_beta_prev = self.L1_beta_star
        if hasattr(self, 'L2_beta_star') and self.L2_beta_star is not None:
            self.L2_beta_prev = self.L2_beta_star

        # Convert to global array
        U_new = self.fitter.segments_to_global(U_new_seg)

        return U_new, U_new_seg, coeffs_new
