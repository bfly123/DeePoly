from typing import Optional, Tuple

import numpy as np


class PseudoTimeFitterAdaptiveMixin:
    def _uses_reference_stable_hysteresis(self) -> bool:
        steady_state_mode = getattr(self.config, "steady_state_mode", True)
        formulation = getattr(self.config, "pseudo_time_formulation", "default")
        return bool(steady_state_mode and formulation == "reference_stable")

    def _get_adaptive_dt_floor(self) -> float:
        """Return a stability-preserving lower bound for adaptive pseudo-time stepping."""
        dt_floor = self.dt_min
        if getattr(self.config, "steady_state_mode", True):
            dt_floor = max(dt_floor, getattr(self, "rollback_dt_min", dt_floor))
        return float(min(dt_floor, self.dt_max))

    def _compute_decrease_rate(
        self,
        residual: float,
        residual_prev: Optional[float],
    ) -> float:
        if residual_prev is not None and residual_prev > 0:
            return residual / residual_prev
        return 1.0

    def _is_metric_improved(self, current_metric: float, best_metric: float) -> bool:
        """Return True when the monitored metric improves enough over the best-so-far state."""
        if not np.isfinite(best_metric):
            return True
        return current_metric < best_metric * self.convergence_improvement_factor

    def _should_restart_on_plateau(
        self,
        current_metric: float,
        no_improvement_count: int,
    ) -> bool:
        """Trigger rollback when the monitored metric plateaus above the acceptable range."""
        if not self.rollback_on_plateau:
            return False
        return (
            no_improvement_count >= self.convergence_plateau_steps
            and current_metric > self.convergence_plateau_tol
        )

    def _get_plateau_restart_dt(self, dt: float) -> float:
        """Shrink dt after a plateau-triggered rollback, following the old solver logic."""
        restart_floor = max(self.rollback_dt_min, self.dt_min)
        return max(dt * self.dt_decrease_factor, restart_floor)

    def _select_next_timestep(
        self,
        dt_current: float,
        residual: float,
        residual_prev: Optional[float],
        stagnation_count: int,
        improved: bool,
        primary_update: float,
    ) -> Tuple[float, int]:
        """Pick the next pseudo-time step using the formulation-specific controller."""
        if self._uses_reference_stable_hysteresis():
            return self._update_reference_stable_timestep(
                dt_current,
                residual,
                residual_prev,
                stagnation_count,
                improved,
                primary_update,
            )
        return self._update_adaptive_timestep(residual, residual_prev, stagnation_count)

    def _update_reference_stable_timestep(
        self,
        dt_current: float,
        residual: float,
        residual_prev: Optional[float],
        stagnation_count: int,
        improved: bool,
        primary_update: float,
    ) -> Tuple[float, int]:
        """Mirror the old stable solver, with optional adaptive tail shrinkage."""
        adaptive_tail = bool(getattr(self.config, "reference_stable_adaptive_tail", False))
        if (
            adaptive_tail
            and self.adaptive_dt
            and primary_update < self.convergence_dt_growth_tol
            and residual < self.ADAPTIVE_DT_THRESHOLD_LOW
        ):
            target_dt, new_stagnation = self._update_adaptive_timestep(
                residual,
                residual_prev,
                stagnation_count,
            )
            staged_dt = max(dt_current * self.dt_decrease_factor, target_dt)
            return float(np.clip(staged_dt, self.dt_min, self.dt_max)), new_stagnation

        next_dt = dt_current
        if improved and primary_update < self.convergence_dt_growth_tol:
            next_dt = min(self.dt_max, dt_current * self.dt_increase_factor)

        new_stagnation = stagnation_count + 1 if residual_prev and residual > residual_prev else 0
        next_dt = float(np.clip(next_dt, self.dt_min, self.dt_max))
        return next_dt, new_stagnation

    def _update_adaptive_timestep(
        self,
        residual: float,
        residual_prev: Optional[float],
        stagnation_count: int,
    ) -> Tuple[float, int]:
        """Update adaptive time step and stagnation count."""
        if not self.adaptive_dt:
            return self.dt_init, stagnation_count

        if residual > self.ADAPTIVE_DT_THRESHOLD_HIGH:
            dt = self.dt_max
        elif residual > self.ADAPTIVE_DT_THRESHOLD_MID:
            dt = self._interpolate_midrange_dt(residual)
        elif residual > self.ADAPTIVE_DT_THRESHOLD_LOW:
            dt = self._interpolate_lowrange_dt(residual)
        else:
            dt = self._get_adaptive_dt_floor()

        dt = float(np.clip(dt, self.dt_min, self.dt_max))
        new_stagnation = stagnation_count + 1 if residual_prev and residual > residual_prev else 0
        return dt, new_stagnation

    def _interpolate_midrange_dt(self, residual: float) -> float:
        dt_range = self.ADAPTIVE_DT_THRESHOLD_HIGH - self.ADAPTIVE_DT_THRESHOLD_MID
        scale = (residual - self.ADAPTIVE_DT_THRESHOLD_MID) / dt_range
        dt_floor = self._get_adaptive_dt_floor()
        dt_target = max(self.dt_max * self.ADAPTIVE_DT_SCALE_FACTOR, dt_floor)
        return dt_target + (self.dt_max - dt_target) * scale

    def _interpolate_lowrange_dt(self, residual: float) -> float:
        dt_range = self.ADAPTIVE_DT_THRESHOLD_MID - self.ADAPTIVE_DT_THRESHOLD_LOW
        scale = (residual - self.ADAPTIVE_DT_THRESHOLD_LOW) / dt_range
        dt_floor = self._get_adaptive_dt_floor()
        dt_target = max(self.dt_max * self.ADAPTIVE_DT_SCALE_FACTOR, dt_floor)
        return dt_floor + (dt_target - dt_floor) * scale
