import numpy as np
from typing import Dict, List, Optional, Tuple

from .linear_solver_backend import CUPY_AVAILABLE
from .trust_region_runtime_mixin import TrustRegionRuntimeMixin
from .trust_region_subproblem_mixin import TrustRegionSubproblemMixin


class TrustRegionSolver(TrustRegionRuntimeMixin, TrustRegionSubproblemMixin):
    """Optimize版信赖域Solve器"""
    
    def __init__(self, use_gpu: bool = True, precondition: bool = True):
        """
        Args:
            use_gpu: YesNoUsingGPU加速
            precondition: YesNoUsing预Process
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.precondition = precondition
        self._last_step = None  # 缓存Up一步Result用于热Start
