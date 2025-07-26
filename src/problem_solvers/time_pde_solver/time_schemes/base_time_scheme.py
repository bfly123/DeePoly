"""Base class for time integration schemes"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class BaseTimeScheme(ABC):
    """抽象基类，定义时间积分格式的通用接口"""
    
    def __init__(self, config):
        self.config = config
        self.fitter = None  # 将由TimePDEFitter设置
    
    def set_fitter(self, fitter):
        """设置fitter引用，用于访问算子和数据"""
        self.fitter = fitter
    
    @abstractmethod
    def time_step(self, u_n: np.ndarray, u_seg: List[np.ndarray], dt: float) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """执行一个时间步
        
        Args:
            u_n: 当前时间步的全局解值
            u_seg: 当前时间步的段级解值列表
            dt: 时间步长
            
        Returns:
            Tuple[np.ndarray, List[np.ndarray], np.ndarray]: (新的全局解值, 新的段级解值, 系数)
        """
        pass
    
    @abstractmethod
    def build_stage_jacobian(self, segment_idx: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """构建时间格式特定的段雅可比矩阵
        
        Args:
            segment_idx: 段索引
            **kwargs: 其他参数（如stage, dt, gamma等）
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (系数矩阵L, 右端向量b)
        """
        pass
    
    @abstractmethod
    def get_scheme_info(self) -> Dict[str, Any]:
        """获取时间格式信息"""
        pass
    
    @abstractmethod
    def validate_operators(self) -> Dict[str, bool]:
        """验证算子配置是否满足时间格式要求"""
        pass
    
    @abstractmethod
    def estimate_stable_dt(self, u_current: np.ndarray, safety_factor: float = 0.8) -> float:
        """估算稳定的时间步长"""
        pass
    
    def print_scheme_summary(self):
        """打印时间格式摘要"""
        validation = self.validate_operators()
        info = self.get_scheme_info()
        
        print(f"=== {info['method']} Time Integration Summary ===")
        print(f"Method: {info['method']}")
        if 'stages' in info:
            print(f"Stages: {info['stages']}, Order: {info.get('order', 'N/A')}")
        if 'gamma' in info:
            print(f"Gamma parameter: {info['gamma']:.6f}")
        print(f"Equation form: {info.get('equation_form', 'N/A')}")
        
        print("\\nOperator Status:")
        for op_name, exists in validation.items():
            if op_name.endswith('_exists'):
                op_display = op_name.replace('_exists', '').upper()
                print(f"  {op_display}: {'✓' if exists else '✗'}")
        
        ready_key = next((k for k in validation.keys() if 'ready' in k), None)
        if ready_key:
            print(f"\\nReady for time integration: {'✓' if validation[ready_key] else '✗'}")
        print("=" * 50)