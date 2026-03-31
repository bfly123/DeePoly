from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from src.abstract_class.config.base_data_custom_loader_mixin import (
    BaseDataCustomLoaderMixin,
)
from src.abstract_class.config.base_data_resolution_mixin import (
    BaseDataResolutionMixin,
)
from src.abstract_class.config.base_data_boundary_mixin import BaseDataBoundaryMixin
from src.abstract_class.config.base_data_segment_mixin import BaseDataSegmentMixin


class BaseDataGenerator(
    BaseDataResolutionMixin,
    BaseDataCustomLoaderMixin,
    BaseDataSegmentMixin,
    BaseDataBoundaryMixin,
    ABC,
):
    def __init__(self, config):
        self.config = config
        self.n_segments = config.n_segments
        self.Ns = np.prod(config.n_segments)
        self.Nw = config.points_per_swap
        self.n_dim = config.n_dim
        self.n_eqs = config.n_eqs
        self._expression_evaluator = self._build_expression_evaluator()
        self.custom_data_generator = self._load_custom_data_generator()

    @abstractmethod
    def generate_global_field(self, x_global: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def generate_data(self, mode: str = "train") -> Dict:
        raise NotImplementedError

    @abstractmethod
    def _prepare_output_dict(self, *args) -> Dict:
        raise NotImplementedError
