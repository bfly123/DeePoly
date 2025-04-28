import numpy as np
import os
import importlib.util
from typing import Dict, Optional
from abstract_class.config.base_data import BaseDataGenerator


class FuncFittingDataGenerator(BaseDataGenerator):
    """函数拟合问题的数据生成器"""

    def __init__(self, config):
        super().__init__(config)
        # 保存配置和案例路径
        self.config = config
        self.case_dir = getattr(config, "case_dir", None)
        # 尝试加载自定义数据生成模块
        self.custom_data_generator = self._load_custom_data_generator()

    def _load_custom_data_generator(self):
        """加载案例目录中的data_generate.py模块"""
        if not self.case_dir:
            print("未设置案例路径，使用默认数据生成方法")
            return None

        data_generate_path = os.path.join(self.case_dir, "data_generate.py")

        if not os.path.exists(data_generate_path):
            print(f"未找到自定义数据生成模块: {data_generate_path}，使用默认数据生成方法")
            return None

        try:
            print(f"加载自定义数据生成模块: {data_generate_path}")
            spec = importlib.util.spec_from_file_location(
                "custom_data_generator", data_generate_path
            )
            data_generator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(data_generator)
            return data_generator
        except Exception as e:
            print(f"加载data_generate.py模块时出错: {e}，将使用默认数据生成方法")
            return None

    def generate_global_field(self, x_global: np.ndarray, mode="train") -> np.ndarray:
        """生成全局场，使用正弦函数创建平滑的分布
        如果存在自定义数据生成器，优先使用自定义方法
        
        Args:
            x_global: 全局点坐标
            mode: 数据模式，"train"或"test"
            
        Returns:
            np.ndarray: 全局场值
        """
        # 如果存在自定义数据生成器，并且实现了generate_global_field方法，则使用它
        if self.custom_data_generator and hasattr(self.custom_data_generator, "generate_global_field"):
            return self.custom_data_generator.generate_global_field(x_global)
        else:
          raise ValueError("未找到自定义数据生成器，或者自定义数据生成器未实现generate_global_field方法")

    def generate_data(self, mode: str = "train") -> Dict:
        """生成训练/测试数据
        如果存在自定义数据生成器，优先使用自定义方法
        """
        # 如果存在自定义数据生成器，并且实现了generate_data方法，则使用它
       # if self.custom_data_generator and hasattr(self.custom_data_generator, "generate_data"):
       #     return self.custom_data_generator.generate_data(x_global)
            
        # 否则使用默认实现
        # 1. 生成全局点和场
        x_global = self._generate_global_points(mode)
        y_global = self.generate_global_field(x_global)

        # 2. 切分到局部段
        x_segments, y_segments = self.split_global_field(x_global, y_global)

        # 3. 处理段数据
        x_swap, x_swap_norm, x_segments_norm, boundary_segments_dict = self._process_segments(x_segments)

        # 4. 准备输出数据
        return self._prepare_output_dict(
            x_segments,
            y_segments,
            x_segments_norm,
            x_swap,
            x_swap_norm,
            boundary_segments_dict
        )
