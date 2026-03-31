from typing import List, Tuple

import numpy as np


class ImexFirstOrderRuntimeMixin:
    def time_step(
        self,
        U_n: np.ndarray,
        U_seg: List[np.ndarray],
        dt: float,
        coeffs_n: np.ndarray = None,
        current_time: float = 0.0,
        step: int = 0,
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        del U_n, coeffs_n
        self._current_time = current_time
        self.U_seg_n = U_seg.copy()
        U_seg_new, coeffs_new = self._solve_imex_stage_U_seg(
            dt=dt,
            stage=1,
            step=step,
            U_n_seg=self.U_seg_n,
        )
        U_new = self.fitter.segments_to_global(U_seg_new)
        return U_new, U_seg_new, coeffs_new

    def _solve_imex_stage_U_seg(self, **kwargs) -> Tuple[List[np.ndarray], np.ndarray]:
        stage_data = {
            "U_n_seg": kwargs.get("U_n_seg"),
            "dt": kwargs.get("dt"),
            "stage": kwargs.get("stage"),
            "operation": "imex_stage",
            "step": kwargs.get("step"),
        }
        coeffs_stage = self.fitter.fit(**stage_data)
        _, U_stage_segments = self.fitter.construct(
            self.fitter.data,
            self.fitter._current_model,
            coeffs_stage,
        )
        return U_stage_segments, coeffs_stage
