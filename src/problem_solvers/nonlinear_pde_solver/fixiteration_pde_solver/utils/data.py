"""Fixed-Point Iteration PDE Data Generator

为稳态PDE生成训练数据,支持源项处理
"""

import numpy as np
from typing import Dict
from ...time_pde_solver.utils.data import TimePDEDataGenerator


class FixIterationPDEDataGenerator(TimePDEDataGenerator):
    """稳态PDE数据生成器

    继承TimePDEDataGenerator,复用空间离散化逻辑
    区别:
    - 不需要时间演化的初始条件
    - 添加源项S的处理
    - 坐标只包含空间维度
    """

    def __init__(self, config):
        super().__init__(config)
        # 验证配置
        if hasattr(config, 'problem_type') and config.problem_type != 'fixiteration_pde':
            print(f"⚠️  警告: FixIterationPDEDataGenerator用于problem_type='fixiteration_pde',"
                  f" 当前为'{config.problem_type}'")

    def generate_data(self, mode: str = "train") -> Dict:
        """生成训练或测试数据

        Args:
            mode: "train" 或 "test"

        Returns:
            data: 数据字典,包含:
                - x: 全局坐标 (n_points, n_dim)
                - x_segments: 分段坐标列表
                - source: 全局源项 (n_points, n_eqs)
                - source_segments: 分段源项列表
                - global_boundary_dict: 边界条件字典
                - ... (其他TimePDEDataGenerator返回的字段)
        """
        # 生成空间点
        x_global = self._generate_global_points(mode)

        # 生成源项 (对于稳态PDE)
        source_term = self._load_source_term(x_global)

        # 处理边界条件
        global_boundary_dict = self.read_boundary_conditions()

        # 分段处理
        x_segments, masks = self.split_global_points(x_global)
        source_segments = self.split_global_field(masks, source_term)

        # 归一化和交换点
        x_segments_norm, x_swap, x_swap_norm, boundary_segments_dict = (
            self._process_segments(x_segments, global_boundary_dict)
        )

        return {
            "x": np.vstack(x_segments),
            "x_segments": x_segments,
            "source": np.vstack(source_segments),          # ✅ 源项
            "source_segments": source_segments,            # ✅ 分段源项
            "global_boundary_dict": global_boundary_dict,
            "x_segments_norm": x_segments_norm,
            "x_swap": x_swap,
            "x_swap_norm": x_swap_norm,
            "boundary_segments_dict": boundary_segments_dict,
        }

    def _load_source_term(self, x_global: np.ndarray) -> np.ndarray:
        """加载源项S

        Args:
            x_global: 全局坐标 (n_points, n_dim)

        Returns:
            source: 源项值 (n_points, n_eqs)
        """
        n_points = x_global.shape[0]
        n_eqs = self.n_eqs

        # 检查配置中是否定义了源项
        if hasattr(self.config, 'eq') and 'S' in self.config.eq:
            S_expressions = self.config.eq['S']

            # 如果所有源项都是"0",返回零数组
            if all(s.strip() == "0" for s in S_expressions):
                return np.zeros((n_points, n_eqs))

            # 否则,需要求值表达式
            source = np.zeros((n_points, n_eqs))
            for eq_idx, expr in enumerate(S_expressions[:n_eqs]):
                if expr.strip() != "0":
                    # 使用与LinearPDESolver相同的方法求值
                    source[:, eq_idx] = self._evaluate_source_expression(expr, x_global)

            return source
        else:
            # 默认零源项
            return np.zeros((n_points, n_eqs))

    def _evaluate_source_expression(self, expr: str, x_global: np.ndarray) -> np.ndarray:
        """求值源项表达式

        Args:
            expr: 源项表达式字符串,如 "sin(pi*x)"
            x_global: 坐标 (n_points, n_dim)

        Returns:
            values: 源项值 (n_points,)
        """
        import sympy as sp
        from src.abstract_class.constants import pi

        # 创建符号变量
        spatial_vars_symbols = {}
        for i, var_name in enumerate(self.config.spatial_vars):
            spatial_vars_symbols[var_name] = sp.Symbol(var_name, real=True)

        # 解析表达式
        try:
            sym_expr = sp.sympify(expr, locals={**spatial_vars_symbols, 'pi': sp.pi})

            # 转为numpy函数
            func = sp.lambdify(
                list(spatial_vars_symbols.values()),
                sym_expr,
                modules=['numpy']
            )

            # 求值
            if self.config.n_dim == 1:
                values = func(x_global[:, 0])
            elif self.config.n_dim == 2:
                values = func(x_global[:, 0], x_global[:, 1])
            elif self.config.n_dim == 3:
                values = func(x_global[:, 0], x_global[:, 1], x_global[:, 2])
            else:
                raise NotImplementedError(f"不支持{self.config.n_dim}维源项求值")

            return values
        except Exception as e:
            print(f"⚠️  源项表达式求值失败: {expr}")
            print(f"   错误: {e}")
            print(f"   使用零源项")
            return np.zeros(x_global.shape[0])

    def generate_global_field(self, x_global: np.ndarray) -> np.ndarray:
        """生成全局场值

        对于稳态PDE,不需要初始条件,返回零场
        (实际初值将由Spotter阶段生成)

        Args:
            x_global: 全局坐标

        Returns:
            zero_field: 零场 (n_points, n_eqs)
        """
        return np.zeros((x_global.shape[0], self.n_eqs))
