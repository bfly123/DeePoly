from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch import nn

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.algebraic_solver import LinearSolver


class LinearPDEFitter(BaseDeepPolyFitter):
    """Linear partial differential equation fitter"""
    
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
        L = self._linear_operators[segment_idx]["L1"]
        
        #eq = []
        #for i in range(self.config.n_eqs):
        #    eq.append(self.equations[f"eq{i}"][segment_idx])
        
        n_points = self.data["x_segments_norm"][segment_idx].shape[0]
        ne = self.n_eqs
        dgN = self.dgN

        L = np.zeros((ne, n_points, ne * dgN))
        b = np.zeros((ne, n_points))

        # construct the fitting equations
        for i in range(ne):
            if source.shape[1] > i:
                b[i,:] = source[:,i]
            else:
                # Handle single variable case where source may only have one column
                b[i,:] = source[:,0]

        # add the spatial discrete terms - placeholder for now
        # This would normally be filled with equation-specific terms
        # For linear PDEs, these are typically handled by the neural network

        # reshape the matrix
        L_reshaped = np.vstack([L[i] for i in range(ne)])
        b_reshaped = np.concatenate([b[i] for i in range(ne)])

        return L_reshaped, b_reshaped