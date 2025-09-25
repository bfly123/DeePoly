from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch import nn
from scipy.sparse import eye, diags
from scipy.sparse.linalg import spsolve

from src.abstract_class.base_fitter import BaseDeepPolyFitter
from src.algebraic_solver import LinearSolver
from ..time_schemes import BaseTimeScheme, create_time_scheme


class TimePDEFitter(BaseDeepPolyFitter):
    """Time-dependent problem hybrid fitter implementation with pluggable time schemes"""

    def __init__(self, config, data: Dict = None, time_scheme: str = "imex_rk_222"):
        super().__init__(config, dt=None, data=data)

        # Initialize solver
        self.solver = LinearSolver(
            verbose=True, use_gpu=True, performance_tracking=True
        )

        # Initialize time integration scheme
        self.time_scheme = self._create_time_scheme(time_scheme)
        self.time_scheme.set_fitter(self)

        # Maintain current segment-level solution values to avoid recomputation
        self.U_seg_current = None  # List[np.ndarray] - current timestep segment values
        self.U_seg_prev = None  # List[np.ndarray] - previous timestep segment values

    def _create_time_scheme(self, scheme_name: str) -> BaseTimeScheme:
        """Create time integration scheme instance using factory pattern"""
        return create_time_scheme(scheme_name, self.config)

    def time_step(self, U_n: np.ndarray, U_seg: List[np.ndarray], dt: float, coeffs_n: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Execute one time step using the configured time scheme
        
        Args:
            U_n: Current timestep global solution values
            U_seg: Current timestep segment-level solution values
            dt: Time step size
            coeffs_n: Current timestep coefficients
            
        Returns:
            Tuple[np.ndarray, List[np.ndarray], np.ndarray]: (new global solution, new segment solutions, coefficients)
        """
        # Update current segment-level solution values
        self.U_seg_current = U_seg
        
        # Delegate to time scheme
        U_new, U_seg_new, coeffs = self.time_scheme.time_step(U_n, U_seg, dt, coeffs_n, **kwargs)
        
        # Update maintained segment-level solution values
        self.U_seg_prev = self.U_seg_current
        self.U_seg_current = U_seg_new
        
        return U_new, U_seg_new, coeffs

    def solve_time_step(self, U_n: np.ndarray, U_seg: List[np.ndarray], dt: float, coeffs_n: np.ndarray = None, **kwargs) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Solve one time step - wrapper for backward compatibility"""
        if not self._precompiled:
            raise RuntimeError("Operators not precompiled. Call fitter_init() first.")
        
        return self.time_step(U_n, U_seg, dt, coeffs_n, **kwargs)

    def get_time_scheme_info(self) -> Dict[str, Any]:
        """Get time integration scheme information"""
        return self.time_scheme.get_scheme_info()

    def validate_operators(self) -> Dict[str, bool]:
        """Validate operator configuration for time integration"""
        return self.time_scheme.validate_operators()

    def estimate_stable_dt(self, U_current: np.ndarray, safety_factor: float = 0.8) -> float:
        """Estimate stable time step size"""
        return self.time_scheme.estimate_stable_dt(U_current, safety_factor)

    def print_time_scheme_summary(self):
        """Print time integration scheme summary"""
        self.time_scheme.print_scheme_summary()



    def _build_segment_jacobian(
        self, segment_idx: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build segment Jacobian matrix using the time scheme
        
        Delegates to the configured time integration scheme for time-specific handling
        """
       # operation = kwargs.get("operation", "imex_stage")
        
       # if operation == "initial_fit":
       #     # 对于初始拟合，使用标准的线性拟合（零右端项）
       #     features = self._features[segment_idx][0]  # 只使用基础特征（u）
       #     ne = self.config.n_eqs
       #     n_points = features.shape[0]
       #     
       #     # 创建单位矩阵作为左端项
       #     L = features
       #     # 创建目标值作为右端项（来自u_seg）
       #     u_seg = kwargs.get("u_seg", [])
       #     if len(u_seg) > segment_idx:
       #         r = u_seg[segment_idx].reshape(-1)  # 展平为1D向量
       #     else:
       #         r = np.zeros(n_points)
       #         
       #     return L, r
       # else:
        return self.time_scheme.build_stage_jacobian(segment_idx, **kwargs)