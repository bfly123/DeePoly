from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from src.abstract_class.config.base_config import BaseConfig
from meta_coding.auto_eq import parse_equation_to_list
import os
import json

@dataclass
class LinearPDEConfig(BaseConfig):
    # 在dataclass中，需要显式声明继承自父类的属性
    case_dir: str  # 从BaseConfig继承的属性
    
    # 添加配置文件中的必要字段
    vars_list: List[str] = field(default_factory=list)  # 变量列表
    spatial_vars: List[str] = field(default_factory=list)  # 空间变量
    eq: List[str] = field(default_factory=list)  # 方程
    eq_nonlinear: List[str] = field(default_factory=list)  # 非线性方程
    const_list: List[str] = field(default_factory=list)  # 常量列表
    source_term: bool = field(default=False)  # 是否包含源项
    
    # 基本参数
    # n_dim通过init=False字段声明，不要在这里重复声明
    n_eqs: int = 1  # 方程数量 (通常是 1)
    n_segments: List[int] = field(default_factory=lambda: [10])  # 分段数量

    # 多项式参数
    poly_degree: List[int] = field(default_factory=lambda: [3])  # 多项式阶数

    # 神经网络参数
    method: str = "hybrid"  # 方法选择：'hybrid' 或 'poly'
    DNN_degree: int = 10  # 神经网络特征维度 (如果 method='hybrid')
    device: str = "cuda"  # 计算设备
    linear_device: str = "cpu"  # 线性求解器设备
    hidden_dims: List[int] = field(
        default_factory=lambda: [64, 64, 64]
    )  # 神经网络隐藏层
    learning_rate: float = 0.001  # 学习率
    max_retries: int = 1  # 最大重试次数
    training_epochs: int = 10000  # 训练轮数

    # 训练/测试数据参数
    n_train: int = 1000  # 训练点总数
    n_test: int = 200  # 测试点总数
    points_domain: int = 1000  # 域内点数
    points_domain_test: int = 200  # 测试域内点数
    points_boundary: int = 200  # 边界点数
    points_boundary_test: int = 50  # 测试边界点数
    points_initial: int = 0  # 初始点数（时间相关问题）

    # 导数参数
    deriv_orders: List[List[int]] = field(
        default_factory=lambda: [[0]]
    )  # 要拟合的导数阶数
    all_derivatives: List[List[int]] = field(
        default_factory=lambda: [[[0]]]
    )  # 界面连续性条件所需的导数阶数 (每 eq)

    # 方程参数
    boundary_conditions: List[dict] = field(default_factory=list)  # 边界条件列表
    
    # PDE特有参数

    # 案例特定参数
    x_domain: List = field(default_factory=list)  # 域边界
    plot_module_path: Optional[str] = None  # 自定义绘图模块的相对路径
    seed: int = 42  # 随机种子
    
    # 定义需要从BaseConfig初始化的字段
    n_dim: int = field(init=False, default=1)  # 问题维度
    x_min: np.ndarray = field(init=False, default=None)  # 每段的最小坐标
    x_max: np.ndarray = field(init=False, default=None)  # 每段的最大坐标
    segment_ranges: List[np.ndarray] = field(
        default_factory=list, init=False
    )  # 每个维度的分段范围
    x_min_norm: np.ndarray = field(init=False, default=None)  # 归一化后的最小坐标
    x_max_norm: np.ndarray = field(init=False, default=None)  # 归一化后的最大坐标
    
    def __post_init__(self):
        """初始化配置参数"""
        BaseConfig.__init__(self, self.case_dir)
        
        self.load_config_from_json(self.case_dir)
        
        # 处理配置文件中的字段映射（例如eq -> Eq）
        self._map_config_fields()
        
        # 验证必要参数
        self._validate_required_params()

        # 初始化其他参数
        self.n_dim = len(self.spatial_vars)
        self.n_eqs = len(self.eq)
        self._init_segment_ranges()
        self._init_boundaries()
        self.init_seed()
        self.DNN_degree = self.hidden_dims[-1]

        # 解析方程
        (
            self.eq_linear_list,
            self.deriv_orders,
            self.max_deriv_orders,
            self.eq_nonlinear_list,
            self.all_derivatives,
        ) = parse_equation_to_list(
            self.eq,
            self.eq_nonlinear,
            self.vars_list,
            self.spatial_vars,
            self.const_list,
        )

    def _map_config_fields(self):
        """处理配置文件中的字段命名与类属性的映射"""
        # 检查 eq 字段是否存在，如果存在则映射到 Eq
        if hasattr(self, 'eq') and not hasattr(self, 'Eq'):
            self.Eq = self.eq
            
        # 检查 eq_nonlinear 字段是否存在，如果存在则映射到 Eq_nonlinear
        if hasattr(self, 'eq_nonlinear') and not hasattr(self, 'Eq_nonlinear'):
            self.Eq_nonlinear = self.eq_nonlinear
            
        # 如果没有设置 const_list，创建空列表
        if not hasattr(self, 'const_list'):
            self.const_list = []

    def _validate_required_params(self):
        """验证必要参数是否已设置"""
        required_params = [
            "vars_list",
            "spatial_vars",
            "n_segments",
            "poly_degree",
            "x_domain",
        ]

        for param in required_params:
            if not hasattr(self, param) or getattr(self, param) is None:
                raise ValueError(f"必要参数 '{param}' 未设置")

        # 特殊验证
        if len(self.spatial_vars) != len(self.n_segments):
            raise ValueError(
                f"spatial_vars长度({len(self.spatial_vars)})和n_segments长度({len(self.n_segments)})不匹配"
            )

        if len(self.spatial_vars) != len(self.poly_degree):
            raise ValueError(
                f"spatial_vars长度({len(self.spatial_vars)})和poly_degree长度({len(self.poly_degree)})不匹配"
            )

    def _init_segment_ranges(self):
        """初始化分段区间"""
        self.segment_ranges = []
        
        for i in range(self.n_dim):
            x_min = self.x_domain[i][0]
            x_max = self.x_domain[i][1]
            n_seg = self.n_segments[i]
            
            # 计算分段区间
            seg_ranges = np.linspace(x_min, x_max, n_seg + 1)
            self.segment_ranges.append(seg_ranges)
            
    def _init_boundaries(self):
        """初始化边界参数"""
        # 创建边界数组
        self.x_min = np.array([range[0] for range in self.segment_ranges])
        self.x_max = np.array([range[-1] for range in self.segment_ranges])
        
        # 归一化边界
        self.x_min_norm = -np.ones(self.n_dim)
        self.x_max_norm = np.ones(self.n_dim)
        
    def init_seed(self):
        """初始化随机种子"""
        np.random.seed(self.seed)
        
    # 覆盖基类方法，定义需要特殊处理的字段
    def _int_list_fields(self):
        """需要转换为整数的字段列表"""
        return ["n_segments", "poly_degree", "hidden_dims"]

    def _list_fields(self):
        """需要特殊处理的列表字段"""
        return ["n_segments", "poly_degree", "hidden_dims", "x_domain"]

    def _process_list_field(self, key, value):
        """处理列表类型字段"""
        if key == "n_segments" or key == "poly_degree" or key == "hidden_dims":
            # 确保是整数列表
            return [int(v) if isinstance(v, str) else v for v in value]
        elif key == "x_domain":
            # 确保是二维列表
            if isinstance(value, list) and len(value) > 0:
                if not isinstance(value[0], list):
                    # 如果是简单的列表[min, max]，转换为[[min, max]]
                    return [value]
            return value
        return value

    def load_config_from_json(self, case_dir=None):
        """从JSON文件加载配置，并更新对象属性
        
        与BaseConfig中不同，此方法会动态添加配置文件中的所有字段，
        即使这些字段未在类中预先定义。

        Args:
            case_dir: 配置文件所在的目录
        """
        config_path = os.path.join(case_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_dict = json.load(f)

                # 将配置字典中的值应用到这个对象
                for key, value in config_dict.items():
                    # 特殊类型处理
                    if isinstance(value, str) and key in self._int_list_fields():
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                    elif isinstance(value, list) and key in self._list_fields():
                        value = self._process_list_field(key, value)

                    # 设置属性，无论是否预先定义
                    setattr(self, key, value)
                    
                print(f"成功从 {config_path} 加载了配置")
                return True
            except Exception as e:
                print(f"加载配置文件时出错: {e}")
                return False
        else:
            print(f"配置文件路径无效: {config_path}")
            return False 