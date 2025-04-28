import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import os

class LinearPDEDataGenerator:
    """线性偏微分方程数据生成器"""
    
    def __init__(self, config):
        """初始化数据生成器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.n_dim = config.n_dim
        self.x_domain = config.x_domain
        
        # 边界条件
        self.boundary_conditions = getattr(config, 'boundary_conditions', [])
        
        # 源项函数（如果有的话）
        self.source_term_func = None
        if config.has_source_term and config.source_term:
            # 将源项字符串转换为函数
            self.source_term_func = self._parse_source_term(config.source_term)
    
    def _parse_source_term(self, source_expr: str) -> Callable:
        """解析源项表达式
        
        Args:
            source_expr: 源项表达式字符串，例如 "sin(pi*x)*sin(pi*y)"
            
        Returns:
            source_func: 源项函数
        """
        # 导入必要的模块
        import numpy as np
        from math import sin, cos, exp, pi, sqrt
        
        # 构建函数表达式
        var_names = ', '.join(self.config.spatial_vars)
        func_str = f"lambda {var_names}: {source_expr}"
        
        try:
            # 编译函数
            source_func = eval(func_str)
            return source_func
        except Exception as e:
            print(f"解析源项表达式时出错: {e}")
            # 返回默认零函数
            return lambda *args: 0.0
    
    def generate_data(self, mode: str = "train") -> Dict:
        """生成训练或测试数据
        
        Args:
            mode: 数据模式，"train" 或 "test"
            
        Returns:
            data: 数据字典，包含 x（坐标）和 u（值）
        """
        if mode == "train":
            n_points = self.config.n_train
            n_domain = self.config.points_domain
            n_boundary = self.config.points_boundary
        else:  # 测试模式
            n_points = self.config.n_test
            n_domain = self.config.points_domain_test
            n_boundary = self.config.points_boundary_test
        
        # 生成域内点
        x_domain, u_domain = self._generate_domain_points(n_domain)
        
        # 生成边界点
        x_boundary, u_boundary = self._generate_boundary_points(n_boundary)
        
        # 组合所有点
        x = np.vstack([x_domain, x_boundary]) if x_boundary.size > 0 else x_domain
        u = np.vstack([u_domain, u_boundary]) if u_boundary.size > 0 else u_domain
        
        # 确保有足够的点
        current_points = x.shape[0]
        if current_points < n_points:
            # 生成额外的随机点
            n_extra = n_points - current_points
            x_extra, u_extra = self._generate_random_points(n_extra)
            x = np.vstack([x, x_extra])
            u = np.vstack([u, u_extra])
        
        # 如果生成的点过多，随机选择n_points个点
        if x.shape[0] > n_points:
            indices = np.random.choice(x.shape[0], n_points, replace=False)
            x = x[indices]
            u = u[indices]
        
        # 准备数据字典
        data = {"x": x, "u": u}
        
        return data
    
    def _generate_domain_points(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成域内点
        
        Args:
            n_points: 点的数量
            
        Returns:
            x: 点坐标 (n_points, n_dim)
            u: 点的值 (n_points, 1)
        """
        # 创建网格或随机点
        # 这里使用随机采样
        x = np.zeros((n_points, self.n_dim))
        
        for dim in range(self.n_dim):
            x_min, x_max = self.x_domain[dim]
            x[:, dim] = np.random.uniform(x_min, x_max, n_points)
        
        # 如果有源项函数，计算u值
        if self.source_term_func:
            u = self._evaluate_source_term(x)
        else:
            # 如果没有源项，默认为零
            u = np.zeros((n_points, 1))
        
        return x, u
    
    def _evaluate_source_term(self, x: np.ndarray) -> np.ndarray:
        """计算源项函数在给定点的值
        
        Args:
            x: 点坐标 (n_points, n_dim)
            
        Returns:
            u: 源项值 (n_points, 1)
        """
        n_points = x.shape[0]
        u = np.zeros((n_points, 1))
        
        # 分别对每个点计算源项值
        for i in range(n_points):
            # 提取当前点的坐标
            point = x[i]
            
            # 计算源项值
            try:
                if self.n_dim == 1:
                    u[i, 0] = self.source_term_func(point[0])
                elif self.n_dim == 2:
                    u[i, 0] = self.source_term_func(point[0], point[1])
                elif self.n_dim == 3:
                    u[i, 0] = self.source_term_func(point[0], point[1], point[2])
                else:
                    # 高维情况
                    args = tuple(point)
                    u[i, 0] = self.source_term_func(*args)
            except Exception as e:
                print(f"计算点 {point} 的源项值时出错: {e}")
                u[i, 0] = 0.0
        
        return u
    
    def _generate_boundary_points(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成边界点
        
        Args:
            n_points: 边界点总数
            
        Returns:
            x: 边界点坐标 (n_boundary_points, n_dim)
            u: 边界点的值 (n_boundary_points, 1)
        """
        if not self.boundary_conditions:
            # 如果没有指定边界条件，返回空数组
            return np.zeros((0, self.n_dim)), np.zeros((0, 1))
        
        # 确定每个边界的点数
        n_boundaries = len(self.boundary_conditions)
        # 将点均匀分配到各个边界，每个边界至少有1个点
        points_per_boundary = max(1, n_points // n_boundaries)
        
        x_boundaries = []
        u_boundaries = []
        
        for bc in self.boundary_conditions:
            # 获取边界信息
            bc_type = bc.get("type", "dirichlet")  # 默认为Dirichlet边界
            bc_value = bc.get("value", 0.0)        # 默认值为0
            bc_location = bc.get("location", {})   # 边界位置定义
            
            # 生成当前边界的点
            x_bc, u_bc = self._generate_points_for_boundary(
                bc_type, bc_value, bc_location, points_per_boundary
            )
            
            if x_bc.size > 0:
                x_boundaries.append(x_bc)
                u_boundaries.append(u_bc)
        
        # 合并所有边界点
        if x_boundaries:
            x = np.vstack(x_boundaries)
            u = np.vstack(u_boundaries)
        else:
            x = np.zeros((0, self.n_dim))
            u = np.zeros((0, 1))
            
        return x, u
    
    def _generate_points_for_boundary(
        self, bc_type: str, bc_value: float, bc_location: Dict, n_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成特定边界的点
        
        Args:
            bc_type: 边界条件类型，"dirichlet"或"neumann"
            bc_value: 边界条件的值
            bc_location: 边界位置定义
            n_points: 点的数量
            
        Returns:
            x: 边界点坐标 (n_points, n_dim)
            u: 边界点的值 (n_points, 1)
        """
        # 初始化坐标数组
        x = np.zeros((n_points, self.n_dim))
        
        # 确定哪个维度是固定的（边界维度）
        fixed_dim = None
        fixed_value = None
        
        for dim, value in bc_location.items():
            try:
                dim_idx = self.config.spatial_vars.index(dim)
                fixed_dim = dim_idx
                fixed_value = value
                break
            except (ValueError, IndexError):
                continue
        
        if fixed_dim is None:
            # 如果没有找到有效的固定维度，返回空数组
            return np.zeros((0, self.n_dim)), np.zeros((0, 1))
        
        # 对每个维度生成坐标
        for dim in range(self.n_dim):
            if dim == fixed_dim:
                # 固定的边界维度
                x[:, dim] = fixed_value
            else:
                # 其他维度随机分布
                x_min, x_max = self.x_domain[dim]
                x[:, dim] = np.random.uniform(x_min, x_max, n_points)
        
        # 根据边界类型设置值
        if bc_type.lower() == "dirichlet":
            # Dirichlet边界：直接设置函数值
            u = np.full((n_points, 1), bc_value)
        elif bc_type.lower() == "neumann":
            # Neumann边界：设置导数值（这里简化处理，仅存储值）
            # 实际应用中，需要在求解过程中特殊处理Neumann条件
            u = np.full((n_points, 1), bc_value)
        else:
            # 未知边界类型，设为0
            u = np.zeros((n_points, 1))
            print(f"警告: 未知的边界条件类型 '{bc_type}'")
        
        return x, u
    
    def _generate_random_points(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """生成随机点
        
        Args:
            n_points: 点的数量
            
        Returns:
            x: 点坐标 (n_points, n_dim)
            u: 点的值 (n_points, 1)
        """
        # 生成随机坐标
        x = np.zeros((n_points, self.n_dim))
        
        for dim in range(self.n_dim):
            x_min, x_max = self.x_domain[dim]
            x[:, dim] = np.random.uniform(x_min, x_max, n_points)
        
        # 初始化值为0
        u = np.zeros((n_points, 1))
        
        return x, u 