from typing import Optional

import numpy as np

from .core.fitter import TimePDEFitter
from .utils.config import TimePDEConfig
from .utils.data import TimePDEDataGenerator
from .utils.visualize import TimePDEVisualizer


class TimeSolverFactoryMixin:
    def _create_config(self, case_dir: str) -> TimePDEConfig:
        """Factory for solver configuration"""
        return TimePDEConfig(case_dir=case_dir)

    def _create_data_generator(self) -> TimePDEDataGenerator:
        """Factory for time-domain data generation"""
        return TimePDEDataGenerator(self.config)

    def _create_model(self):
        """Factory for the neural network model"""
        if not self._needs_live_torch_model():
            return self._build_external_only_spotter_model()
        from .core.net import TimePDENet

        return TimePDENet(self.config)

    def _create_fitter(self, time_scheme: Optional[str]) -> TimePDEFitter:
        """Factory for the time-marching fitter"""
        return TimePDEFitter(config=self.config, data=self.data_train, time_scheme=time_scheme)

    def _create_visualizer(self) -> TimePDEVisualizer:
        """Factory for visualization utilities"""
        return TimePDEVisualizer(self.config)

    def _get_default_time_scheme(self) -> str:
        """Default legacy-compatible time scheme used when config omits one"""
        return "onestep_predictor"

    def _get_spotter_net_class(self):
        """Provide SpotterNet class for optional spotter initialization"""
        try:
            from ..pseudotime_pde_solver.core.spotter_init import SpotterNet

            return SpotterNet
        except ImportError as exc:
            raise NotImplementedError("SpotterNet not available for TimePDESolver") from exc
