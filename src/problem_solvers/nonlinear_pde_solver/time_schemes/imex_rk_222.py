"""IMEX-RK(2,2,2) time integration scheme"""

import numpy as np

from .base_time_scheme import BaseTimeScheme
from .imex_rk_debug_mixin import ImexRKDebugMixin
from .imex_rk_stage_mixin import ImexRKStageMixin


class ImexRK222(ImexRKDebugMixin, ImexRKStageMixin, BaseTimeScheme):
    """IMEX-RK(2,2,2) time integration scheme implementation"""

    def __init__(self, config):
        super().__init__(config)
        self.gamma = (2 - np.sqrt(2)) / 2
        self.order = 2
        self.A_imp = np.array([[self.gamma, 0], [1 - 2 * self.gamma, self.gamma]])
        self.A_exp = np.array([[0, 0], [1 - self.gamma, 0]])
        self.b = np.array([0.5, 0.5])
        self.stage_solutions = {}
        self.U_seg_current = None
        self.U_seg_prev = None
