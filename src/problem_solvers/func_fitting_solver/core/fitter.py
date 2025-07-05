from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch import nn

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.algebraic_solver import LinearSolver

class FuncFittingFitter(BaseDeepPolyFitter):
    """Mixed fitter implementation for function fitting problems"""

    def __init__(self, config, data: Dict = None):
        super().__init__(config, data)
        self.data = data
        # Initialize solver
        self.solver = LinearSolver(verbose=True, use_gpu=True, performance_tracking=True)

    def get_segment_data(self, segment_idx: int) -> Dict:
        """Get data for specified segment"""
        return {
            "x": self.data["x_segments"][segment_idx],
            "u": self.data["u_segments"][segment_idx]
        }

    def _build_segment_jacobian(
        self,
        segment_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build Jacobian matrix for single segment"""
        # Get data
        u_target = self.data["u_segments"][segment_idx]
        
        eq = []
        for i in range(self.config.n_eqs):
            eq.append(self.equations[f"eq{i}"][segment_idx])
        
        n_points = self.data["x_segments_norm"][segment_idx].shape[0]
        ne = self.n_eqs
        dgN = self.dgN

        L = np.zeros((ne, n_points, ne * dgN))
        b = np.zeros((ne, n_points))

        # Build fitting equations - using target function values
        for i in range(ne):
            b[i,:] = u_target[:,i]

        # Add spatial discretization terms
        for i in range(ne):
            L[i] = eq[i]

        # Reshape matrices
        L = np.vstack([L[i] for i in range(ne)])
        b = np.vstack([b[i].reshape(-1, 1) for i in range(ne)])

        return L, b
