from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch import nn

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.algebraic_solver import LinearSolver

class FuncFittingFitter(BaseDeepPolyFitter):
    """Function fitting fitter implementation"""

    def __init__(self, config, data: Dict = None):
        super().__init__(config, data)
        self.data = data
        self.solver = LinearSolver(verbose=True, use_gpu=True, performance_tracking=True)

    def _build_segment_jacobian(
        self,
        segment_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build Jacobian matrix for single segment"""
        u_target = self.data["u_segments"][segment_idx]
        L = self._linear_operators[segment_idx]["L1"]
        
        n_points = self.data["x_segments_norm"][segment_idx].shape[0]
        ne = self.n_eqs

        b = np.zeros((ne, n_points))

        # Build fitting equations using target function values
        for i in range(ne):
            b[i,:] = u_target[:,i]

        # Reshape matrices
        L = np.vstack([L[i] for i in range(ne)])
        b = np.vstack([b[i].reshape(-1, 1) for i in range(ne)])

        return L, b
