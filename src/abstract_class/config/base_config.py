from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
import numpy as np
import torch
import math
import json
import os


@dataclass
class BaseConfig(ABC):
    """配置基类，用于定义所有配置类的基本接口和通用功能"""

    def __init__(self, case_dir: str):
        self.case_dir = case_dir

    # 运行时计算的字段
    n_dim: int = field(init=False)  # 问题维度
    x_min: np.ndarray = field(init=False)  # 每段的最小坐标
    x_max: np.ndarray = field(init=False)  # 每段的最大坐标
    segment_ranges: List[np.ndarray] = field(
        default_factory=list, init=False
    )  # 每个维度的分段范围
    x_min_norm: np.ndarray = field(init=False)  # 归一化后的最小坐标
    x_max_norm: np.ndarray = field(init=False)  # 归一化后的最大坐标

    @abstractmethod
    def __post_init__(self):
        """初始化配置参数，需要在子类中实现"""
        pass

    def _init_segment_ranges(self):
        """初始化分段范围，支持任意维度"""
        self.segment_ranges = [
            np.linspace(
                self.x_domain[dim][0], self.x_domain[dim][1], self.n_segments[dim] + 1
            )
            for dim in range(self.n_dim)
        ]
        self.x_domain = np.array(self.x_domain)
        self.poly_degree = np.array(self.poly_degree)

    def _init_boundaries(self):
        """初始化边界值，支持任意维度"""
        # 计算总段数
        Ns = math.prod(self.n_segments)

        # 初始化数组
        self.x_min = np.zeros((Ns, self.n_dim))
        self.x_max = np.zeros((Ns, self.n_dim))
        self.x_min_norm = np.zeros((Ns, self.n_dim))
        self.x_max_norm = np.ones((Ns, self.n_dim))

        # 为每个段设置边界
        for n in range(Ns):
            # 将段索引 n 转换为多维索引
            indices = self._segment_index_to_multi_index(n, self.n_segments)

            # 设置每个维度的边界
            for dim in range(self.n_dim):
                self.x_min[n, dim] = self.segment_ranges[dim][indices[dim]]
                self.x_max[n, dim] = self.segment_ranges[dim][indices[dim] + 1]

    def _segment_index_to_multi_index(
        self, index: int, n_segments: List[int]
    ) -> List[int]:
        """将一维段索引转换为多维索引

        Args:
            index: 一维段索引
            n_segments: 每个维度的段数

        Returns:
            List[int]: 多维索引
        """
        multi_index = []
        remaining = index

        # 从最后一个维度开始，向第一个维度推进
        for i in range(self.n_dim - 1, -1, -1):
            # 计算当前维度前所有维度的乘积
            divisor = 1
            for j in range(i):
                divisor *= n_segments[j]

            # 计算当前维度的索引
            dim_index = remaining // divisor
            remaining %= divisor

            # 在列表前方插入此维度索引
            multi_index.insert(0, dim_index)

        return multi_index

    @property
    def device(self) -> str:
        """返回计算设备"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def init_seed(self, seed: int = None):
        """初始化随机种子

        Args:
            seed: 随机种子值，如果为None则使用配置中的seed
        """
        seed = seed if seed is not None else self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    def export_to_json(self, filename="config.json"):
        """导出配置到JSON文件

        Args:
            filename: 输出JSON文件名
        """

        def serialize(obj):
            """辅助函数，用于序列化默认不可JSON序列化的对象"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (tuple, list)):
                return [serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: serialize(value) for key, value in obj.items()}
            elif hasattr(obj, "__dict__"):
                return serialize(obj.__dict__)
            else:
                return str(obj)

        # 获取所有非方法和非私有的属性
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and not callable(value):
                config_dict[key] = value

        # 添加元数据
        config_dict["export_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_dict["device_info"] = self.device

        # 序列化并保存到文件
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(serialize(config_dict), f, indent=2)

        print(f"配置已导出到 {filename}")

    def load_config_from_json(self, case_dir=None):
        """从JSON文件加载配置

        Args:
            config_path: 配置文件路径，默认为当前目录下的config.json
        """
        config_path = os.path.join(case_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config_dict = json.load(f)

                # 将配置字典中的值应用到这个对象
                for key, value in config_dict.items():
                    if hasattr(self, key):
                        # 特殊类型处理
                        if isinstance(value, str) and key in self._int_list_fields():
                            try:
                                value = int(value)
                            except ValueError:
                                pass
                        elif isinstance(value, list) and key in self._list_fields():
                            value = self._process_list_field(key, value)

                        setattr(self, key, value)
                return True
            except Exception as e:
                print(f"加载配置文件时出错: {e}")
                return False
        return False

    def _int_list_fields(self):
        """需要转换为整数的字段列表"""
        return []

    def _list_fields(self):
        """需要特殊处理的列表字段"""
        return []

    def _process_list_field(self, key, value):
        """处理列表类型字段"""
        return value
