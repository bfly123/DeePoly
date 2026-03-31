"""Fixed-point fitter facade with split setup and Jacobian helpers."""

from typing import Dict

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from .fixiteration_fitter_jacobian_mixin import FixIterationFitterJacobianMixin
from .fixiteration_fitter_setup_mixin import FixIterationFitterSetupMixin


class FixIterationPDEFitter(
    FixIterationFitterJacobianMixin,
    FixIterationFitterSetupMixin,
    BaseDeepPolyFitter,
):
    """Fixed-point iteration fitter for steady-state PDEs

    Core idea: Fix nonlinear term N(U^n), solve linear system for U^{n+1}
    """

    def __init__(self, config, data: Dict = None):
        super().__init__(config, data=data)
        self._initialize_fixiteration_runtime(config, data=data)
