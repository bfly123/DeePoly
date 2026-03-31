"""IMEX First-order (Backward Euler) time integration scheme"""

from typing import List

import numpy as np

from .base_time_scheme import BaseTimeScheme
from .imex_first_order_jacobian_mixin import ImexFirstOrderJacobianMixin
from .imex_first_order_runtime_mixin import ImexFirstOrderRuntimeMixin
from .imex_first_order_support_mixin import ImexFirstOrderSupportMixin


class ImexFirstOrder(
    ImexFirstOrderRuntimeMixin,
    ImexFirstOrderJacobianMixin,
    ImexFirstOrderSupportMixin,
    BaseTimeScheme,
):
    """IMEX First-order (Backward Euler) time integration scheme implementation"""

    def __init__(self, config):
        super().__init__(config)
        self.gamma = 1.0
        self.order = 1
        self.A_imp = np.array([[1.0]])
        self.A_exp = np.array([[0.0]])
        self.b = np.array([1.0])
        self.stage_solutions = {}
        self.U_seg_current = None
        self.U_seg_prev = None
