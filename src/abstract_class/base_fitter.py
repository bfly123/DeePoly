from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

from .base_fitter_constraints_mixin import BaseFitterConstraintsMixin
from .base_fitter_core_mixin import BaseFitterCoreMixin
from .base_fitter_prediction_mixin import BaseFitterPredictionMixin
from .spotter_feature_provider import resolve_spotter_feature_dim


class BaseDeepPolyFitter(
    BaseFitterPredictionMixin,
    BaseFitterConstraintsMixin,
    BaseFitterCoreMixin,
    ABC,
):
    """Abstract base class for DeePoly fitters"""

    def __init__(self, config, dt: float = None, data: Dict = None, time_scheme: str = None):
        from src.abstract_class.features_generator import FeatureGenerator

        self.config = config
        self.data = data
        self.dt = dt
        self.time_scheme = None
        self._time_scheme_name = time_scheme
        if time_scheme is not None:
            self._initialize_time_scheme(time_scheme)

        self.U_seg_current = None
        self.U_seg_prev = None
        self.n_dim = config.n_dim
        raw_segments = getattr(config, "n_segments", 10)
        self.n_segments = raw_segments if isinstance(raw_segments, (list, tuple)) else [raw_segments] * self.n_dim
        self.ns = np.prod(self.n_segments)

        self._features = {}
        self._linear_operators = {}
        self._nonlinear_functions = {}
        self._precompiled = False
        self.feature_generator = FeatureGenerator(self.config.linear_device)

        operator_source = getattr(config, "operator_parse", {})

        def safe_get(source, key, default=None):
            if hasattr(source, "get"):
                return source.get(key, default)
            if hasattr(source, key):
                return getattr(source, key)
            return default

        self.max_derivatives = safe_get(operator_source, "max_derivative_orders") or getattr(config, "max_derivative_orders", [])
        self.all_derivatives = safe_get(operator_source, "all_derivatives") or getattr(config, "all_derivatives", {})
        self.derivatives = safe_get(operator_source, "derivatives") or getattr(config, "derivatives", [])
        self.operator_terms = safe_get(operator_source, "operator_terms") or getattr(config, "operator_terms", {})
        self.L1 = self.operator_terms.get("L1", None)
        self.L2 = self.operator_terms.get("L2", None)
        self.N = self.operator_terms.get("N", None)
        self.F = self.operator_terms.get("F", None)

        poly_degree = np.asarray(self.config.poly_degree)
        self.dg = np.int32(np.prod(poly_degree + 1))
        self.nn_feature_dim = int(getattr(self.config, "DNN_degree", 0)) if self.config.method == "hybrid" else 0
        self.dgN = int(self.dg) + int(self.nn_feature_dim)
        self.device = self.config.linear_device
        self.n_eqs = self.config.n_eqs
        self.A_constraints = None
        self.b_constraints = None
        self.A = None
        self.b = None
        self._generate_operator_functions()

    def _update_feature_dimensions(self, feature_source) -> None:
        if getattr(self.config, "method", "hybrid") != "hybrid":
            self.nn_feature_dim = 0
            self.dgN = int(self.dg)
            return

        sample_x_norm = None
        x_min = None
        x_max = None
        x_segments_norm = self.data.get("x_segments_norm") if isinstance(self.data, dict) else None
        if x_segments_norm:
            for segment_idx, x_seg in enumerate(x_segments_norm):
                if x_seg is None or len(x_seg) == 0:
                    continue
                sample_x_norm = np.asarray(x_seg[:1], dtype=np.float64)
                x_min = np.asarray(self.config.x_min[segment_idx], dtype=np.float64)
                x_max = np.asarray(self.config.x_max[segment_idx], dtype=np.float64)
                break

        if sample_x_norm is None or x_min is None or x_max is None:
            feature_dim = int(getattr(self.config, "DNN_degree", 0))
        else:
            feature_dim = resolve_spotter_feature_dim(
                feature_source,
                sample_x_norm=sample_x_norm,
                x_min=x_min,
                x_max=x_max,
                device=getattr(self.config, "device", "cpu"),
            )

        self.nn_feature_dim = int(feature_dim)
        self.dgN = int(self.dg) + int(self.nn_feature_dim)

    @abstractmethod
    def _build_segment_jacobian(self, segment_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build Jacobian matrix for single segment"""
        pass
