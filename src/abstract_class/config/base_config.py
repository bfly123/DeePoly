from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
import math

import numpy as np

from src.abstract_class.config.base_config_json_mixin import BaseConfigJsonMixin


@dataclass
class BaseConfig(BaseConfigJsonMixin, ABC):
    """Base interface and shared runtime helpers for solver configurations."""

    def __init__(self, case_dir: str):
        self.case_dir = case_dir

    n_dim: int = field(init=False)
    x_min: np.ndarray = field(init=False)
    x_max: np.ndarray = field(init=False)
    segment_ranges: List[np.ndarray] = field(default_factory=list, init=False)
    x_min_norm: np.ndarray = field(init=False)
    x_max_norm: np.ndarray = field(init=False)

    @abstractmethod
    def __post_init__(self):
        raise NotImplementedError

    def get_unified_attribute(self, attr_name: str, fallback_source: str = None, default=None):
        if hasattr(self, attr_name):
            return getattr(self, attr_name)
        if fallback_source and hasattr(self, fallback_source):
            fallback_dict = getattr(self, fallback_source)
            if isinstance(fallback_dict, dict) and attr_name in fallback_dict:
                return fallback_dict[attr_name]
        return default

    def _init_segment_ranges(self):
        self.segment_ranges = [
            np.linspace(
                self.x_domain[dim][0], self.x_domain[dim][1], self.n_segments[dim] + 1
            )
            for dim in range(self.n_dim)
        ]
        self.x_domain = np.array(self.x_domain)
        self.poly_degree = np.array(self.poly_degree)

    def _init_boundaries(self):
        Ns = math.prod(self.n_segments)
        self.x_min = np.zeros((Ns, self.n_dim))
        self.x_max = np.zeros((Ns, self.n_dim))
        self.x_min_norm = np.zeros((Ns, self.n_dim))
        self.x_max_norm = np.ones((Ns, self.n_dim))
        for n in range(Ns):
            indices = np.unravel_index(n, self.n_segments, order="F")
            for dim in range(self.n_dim):
                self.x_min[n, dim] = self.segment_ranges[dim][indices[dim]]
                self.x_max[n, dim] = self.segment_ranges[dim][indices[dim] + 1]

    @property
    def device(self) -> str:
        torch = self._torch_module()
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    def init_seed(self, seed: int = None):
        seed = seed if seed is not None else self.seed
        np.random.seed(seed)
        torch = self._torch_module()
        if torch is None:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    def _int_list_fields(self):
        return []

    def _list_fields(self):
        return []

    def _process_list_field(self, key, value):
        return value
    @staticmethod
    def _torch_module():
        try:
            import torch
        except ImportError:
            return None
        return torch
