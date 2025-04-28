from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
from meta_coding.auto_eq import parse_equation_to_list
from meta_coding.auto_repalce_nonlinear import update_hybrid_fitter_code
from meta_coding.auto_replace_loss import update_physics_loss_code
from abstract_class.config.base_config import BaseConfig

@dataclass
class TimePDEConfig(BaseConfig):
    """时间依赖问题的配置类"""

    # 方程相关配置
    problem_type: str = "TimePDE"  # 问题类型
    Eq: List[str] = field(default_factory=lambda: [
        "diff(u,x)+diff(u,y)"
    ])
    vars_list: List[str] = field(default_factory=lambda: ["u"])
    spatial_vars: List[str] = field(default_factory=lambda: ["x", "y"])
    const_list: List[Dict[str, float]] = field(default_factory=lambda: [{"Re": 3200, "nu": 0.01}])
    Eq_nonlinear: List[str] = field(default_factory=lambda: ["u*u"])
    re_coding: bool = False  # 是否重新编码
    Stable: bool = True  # 是否稳定
    dt: float = 0.05  # 时间步长
    time: float = 0.3  # 时间

    # BDF2方法参数
    gamma_bdf2: float = 2-np.sqrt(2)
    theta_bdf2: float = 1-0.5/(2-np.sqrt(2))

    # 运行时计算的字段
    n_eqs: int = field(init=False)  # 方程数量
    eq_linear_list: List[str] = field(default_factory=list)
    deriv_orders: List[int] = field(default_factory=list)
    max_deriv_orders: List[int] = field(default_factory=list)
    eq_nonlinear_list: List[str] = field(default_factory=list)
    all_derivatives: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        """初始化配置参数"""
        super().__post_init__()
        self.n_dim = len(self.spatial_vars)  # 初始化问题维度
        self.n_eqs = len(self.Eq)  # 初始化方程数量
        self._init_segment_ranges()
        self._init_boundaries()
        self.init_seed()
        self.DNN_degree = self.hidden_dims[-1]

        if self.re_coding:
            update_hybrid_fitter_code(self.Eq_nonlinear, self.vars_list, self.spatial_vars)
            update_physics_loss_code(
                self.Eq, 
                self.Eq_nonlinear, 
                self.vars_list, 
                self.spatial_vars, 
                self.const_list
            )

        self.eq_linear_list, self.deriv_orders, self.max_deriv_orders, self.eq_nonlinear_list, self.all_derivatives = parse_equation_to_list(
            self.Eq, self.Eq_nonlinear, self.vars_list, self.spatial_vars, self.const_list
        )
