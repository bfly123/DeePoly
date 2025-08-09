"""Base class for time integration schemes"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class BaseTimeScheme(ABC):
    """Abstract base class defining common interface for time integration schemes"""
    
    def __init__(self, config):
        self.config = config
        self.fitter = None  # 将由TimePDEFitter设置
    
    def set_fitter(self, fitter):
        """Set fitter reference for accessing operators and data"""
        self.fitter = fitter
    
    @abstractmethod
    def time_step(self, u_n: np.ndarray, u_seg: List[np.ndarray], dt: float) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """Execute one time step
        
        Args:
            u_n: Current timestep global solution values
            u_seg: Current timestep segment-level solution values list
            dt: Time step size
            
        Returns:
            Tuple[np.ndarray, List[np.ndarray], np.ndarray]: (new global solution values, new segment solution values, coefficients)
        """
        pass
    
    @abstractmethod
    def build_stage_jacobian(self, segment_idx: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Build time scheme-specific segment Jacobian matrix
        
        Args:
            segment_idx: Segment index
            **kwargs: Other parameters (such as stage, dt, gamma, etc.)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (coefficient matrix L, right-hand side vector b)
        """
        pass
    
    @abstractmethod
    def get_scheme_info(self) -> Dict[str, Any]:
        """Get time integration scheme information"""
        pass
    
    @abstractmethod
    def validate_operators(self) -> Dict[str, bool]:
        """Validate whether operator configuration meets time scheme requirements"""
        pass
    
    @abstractmethod
    def estimate_stable_dt(self, u_current: np.ndarray, safety_factor: float = 0.8) -> float:
        """Estimate stable time step size"""
        pass
    
    def print_scheme_summary(self):
        """Print time integration scheme summary"""
        validation = self.validate_operators()
        info = self.get_scheme_info()
        
        print(f"=== {info['method']} Time Integration Summary ===")
        print(f"Method: {info['method']}")
        if 'stages' in info:
            print(f"Stages: {info['stages']}, Order: {info.get('order', 'N/A')}")
        if 'gamma' in info:
            print(f"Gamma parameter: {info['gamma']:.6f}")
        print(f"Equation form: {info.get('equation_form', 'N/A')}")
        
        print("\nOperator Status:")
        for op_name, exists in validation.items():
            if op_name.endswith('_exists'):
                op_display = op_name.replace('_exists', '').upper()
                print(f"  {op_display}: {'✓' if exists else '✗'}")
        
        ready_key = next((k for k in validation.keys() if 'ready' in k), None)
        if ready_key:
            print(f"\nReady for time integration: {'✓' if validation[ready_key] else '✗'}")
        print("=" * 50)