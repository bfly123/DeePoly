from typing import Any
from .feature_generator_nn_mixin import FeatureGeneratorNNMixin
from .feature_generator_poly_mixin import FeatureGeneratorPolyMixin


class FeatureGenerator(FeatureGeneratorNNMixin, FeatureGeneratorPolyMixin):
    """Feature generator class for polynomial and neural network features"""

    def __init__(self, device: Any):
        self.device = device
