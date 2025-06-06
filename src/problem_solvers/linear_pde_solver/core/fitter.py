from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch import nn

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.algebraic_solver import LinearSolver


class LinearPDEFitter(BaseDeepPolyFitter):
    """线性偏微分方程拟合器"""
    
    def __init__(self, config, data: Dict = None):
        super().__init__(config, data)
        self.data = data
        # initialize the solver
        self.solver = LinearSolver(verbose=True, use_gpu=True, performance_tracking=True)

#    def get_segment_data(self, segment_idx: int) -> Dict:
#        """get the data for each segment"""
#        return {
#            "x_norm": self.data["x_segments_norm"][segment_idx],
#            "source": self.data["source_segments"][segment_idx],
#            "boundary_segments_dict": self.data["boundary_segments_dict"][segment_idx]
#        }

    def _build_segment_jacobian(
        self,
        segment_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """construct the jacobian matrix for each segment"""
        # get the data
        source = self.data["source_segments"][segment_idx]
        
        eq = []
        for i in range(self.config.n_eqs):
            eq.append(self.equations[f"eq{i}"][segment_idx])
        
        n_points = self.data["x_segments_norm"][segment_idx].shape[0]
        ne = self.n_eqs
        dgN = self.dgN

        L = np.zeros((ne, n_points, ne * dgN))
        b = np.zeros((ne, n_points))

        # construct the fitting equations
        for i in range(ne):
            b[i,:] = source[:,i]

        # add the spatial discrete terms
        for i in range(ne):
            L[i] = eq[i]

        # reshape the matrix
        L = np.vstack([L[i] for i in range(ne)])
        b = np.vstack([b[i].reshape(-1, 1) for i in range(ne)])

        return L, b