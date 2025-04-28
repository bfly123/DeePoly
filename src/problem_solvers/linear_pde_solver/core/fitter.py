import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any

class LinearPDEFitter:
    """线性偏微分方程拟合器"""
    
    def __init__(self, config, train_data: Dict):
        """初始化拟合器
        
        Args:
            config: 配置对象
            train_data: 训练数据字典
        """
        self.config = config
        self.train_data = train_data
        self.n_dim = config.n_dim
        self.n_segments = config.n_segments
        
        # 初始化分段相关变量
        self.segment_ranges = getattr(config, "segment_ranges", None)
        if self.segment_ranges is None:
            self._init_segment_ranges()
            
        # 多项式度数
        self.poly_degree = config.poly_degree
        
    def _init_segment_ranges(self):
        """初始化分段区间"""
        self.segment_ranges = []
        for dim in range(self.n_dim):
            # 获取当前维度的边界
            x_min = self.config.x_domain[dim][0]
            x_max = self.config.x_domain[dim][1]
            n_seg = self.config.n_segments[dim]
            
            # 计算分段区间
            segment_range = np.linspace(x_min, x_max, n_seg + 1)
            self.segment_ranges.append(segment_range)
    
    def fitter_init(self, model):
        """初始化拟合过程
        
        Args:
            model: 神经网络模型
        """
        self.model = model
        # 可以在这里添加更多初始化逻辑
    
    def fit(self) -> np.ndarray:
        """执行拟合过程，求解方程的系数
        
        Returns:
            coeffs: 方程系数数组
        """
        # 将神经网络设置为评估模式
        self.model.eval()
        
        # 创建系数容器
        total_segments = np.prod(self.config.n_segments)
        # 每个分段中每个维度的多项式系数数量
        poly_terms_per_dim = [self.poly_degree[i] + 1 for i in range(self.n_dim)]
        total_coeffs = np.prod(poly_terms_per_dim) * total_segments
        
        # 初始化系数数组
        coeffs = np.zeros(total_coeffs)
        
        print(f"开始拟合，总系数数量: {total_coeffs}")
        
        # 生成分段拟合所需数据
        with torch.no_grad():
            # 获取训练点
            x_train = self.train_data["x"]
            
            # 使用神经网络生成预测值
            x_train_tensor = torch.tensor(
                x_train, dtype=torch.float32, device=self.config.device
            )
            pred_train = self.model(x_train_tensor).cpu().numpy()
            
            # 将预测值保存到训练数据中
            self.train_data["u_pred"] = pred_train
            
            # 根据神经网络预测值拟合多项式
            coeffs = self._fit_polynomial(x_train, pred_train)
        
        return coeffs
    
    def _fit_polynomial(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """使用多项式拟合每个分段
        
        Args:
            x: 输入坐标 (n_points, n_dim)
            u: 目标值 (n_points, 1)
            
        Returns:
            coeffs: 拟合的多项式系数
        """
        # 获取数据点数量
        n_points = x.shape[0]
        
        # 初始化系数容器
        total_segments = np.prod(self.config.n_segments)
        poly_terms_per_dim = [self.poly_degree[i] + 1 for i in range(self.n_dim)]
        coeffs_per_segment = np.prod(poly_terms_per_dim)
        coeffs = np.zeros(coeffs_per_segment * total_segments)
        
        # 为每个分段拟合多项式
        segment_idx = 0
        segment_indices = [0] * self.n_dim
        
        # 递归拟合每个分段
        self._fit_segment_recursive(
            x, u, 0, segment_indices, segment_idx, coeffs
        )
        
        return coeffs
    
    def _fit_segment_recursive(
        self, x: np.ndarray, u: np.ndarray, dim: int, 
        segment_indices: List[int], segment_idx: int, 
        coeffs: np.ndarray
    ):
        """递归地拟合每个分段
        
        Args:
            x: 输入坐标 (n_points, n_dim)
            u: 目标值 (n_points, 1)
            dim: 当前维度
            segment_indices: 各维度的分段索引
            segment_idx: 当前分段全局索引
            coeffs: 系数数组（将被修改）
        """
        if dim == self.n_dim:
            # 递归结束，开始拟合当前分段
            
            # 计算当前分段的边界
            segment_bounds = []
            for d in range(self.n_dim):
                idx = segment_indices[d]
                segment_bounds.append((
                    self.segment_ranges[d][idx],
                    self.segment_ranges[d][idx + 1]
                ))
            
            # 选择当前分段内的点
            mask = np.ones(x.shape[0], dtype=bool)
            for d in range(self.n_dim):
                x_d = x[:, d]
                low, high = segment_bounds[d]
                if d < self.n_dim - 1:
                    # 对于除最后一个维度外的所有维度，使用半开区间 [low, high)
                    dim_mask = (x_d >= low) & (x_d < high)
                else:
                    # 对最后一个维度使用闭区间 [low, high]，确保边界点被包含
                    dim_mask = (x_d >= low) & (x_d <= high)
                mask = mask & dim_mask
                
            # 如果分段中没有点，跳过拟合
            if np.sum(mask) == 0:
                print(f"警告: 分段 {segment_indices} 中没有数据点")
                return
                
            # 提取当前分段中的点
            x_segment = x[mask]
            u_segment = u[mask]
            
            # 计算系数在全局系数数组中的起始索引
            poly_terms_per_dim = [self.poly_degree[i] + 1 for i in range(self.n_dim)]
            coeffs_per_segment = np.prod(poly_terms_per_dim)
            start_idx = segment_idx * coeffs_per_segment
            
            # 执行多项式拟合
            segment_coeffs = self._fit_segment_poly(
                x_segment, u_segment, segment_bounds
            )
            
            # 将拟合系数存储到全局系数数组
            coeffs[start_idx:start_idx + coeffs_per_segment] = segment_coeffs.flatten()
            
        else:
            # 继续递归到下一个维度
            for i in range(self.config.n_segments[dim]):
                segment_indices[dim] = i
                
                # 计算全局段索引
                if dim == 0:
                    current_segment_idx = i
                else:
                    # 计算当前维度对全局索引的贡献
                    dim_contribution = i
                    for d in range(dim):
                        dim_contribution *= self.config.n_segments[d]
                    current_segment_idx = segment_idx + dim_contribution
                
                # 递归到下一维度
                self._fit_segment_recursive(
                    x, u, dim + 1, segment_indices, current_segment_idx, coeffs
                )
    
    def _fit_segment_poly(
        self, x_segment: np.ndarray, u_segment: np.ndarray, 
        segment_bounds: List[Tuple[float, float]]
    ) -> np.ndarray:
        """对单个分段执行多项式拟合
        
        Args:
            x_segment: 分段内的输入坐标 (n_segment_points, n_dim)
            u_segment: 分段内的目标值 (n_segment_points, 1)
            segment_bounds: 分段边界 [(x1_min, x1_max), (x2_min, x2_max), ...]
            
        Returns:
            segment_coeffs: 分段的多项式系数数组
        """
        # 计算归一化坐标
        x_norm = np.zeros_like(x_segment)
        for d in range(self.n_dim):
            low, high = segment_bounds[d]
            x_norm[:, d] = 2.0 * (x_segment[:, d] - low) / (high - low) - 1.0
            
        # 构建多项式设计矩阵
        poly_terms_per_dim = [self.poly_degree[i] + 1 for i in range(self.n_dim)]
        
        # 对于一维情况的简单处理
        if self.n_dim == 1:
            design_matrix = np.ones((x_segment.shape[0], poly_terms_per_dim[0]))
            for j in range(1, poly_terms_per_dim[0]):
                design_matrix[:, j] = design_matrix[:, j-1] * x_norm[:, 0]
                
            # 使用最小二乘法求解多项式系数
            segment_coeffs, _, _, _ = np.linalg.lstsq(design_matrix, u_segment, rcond=None)
            
        else:
            # 对于多维情况
            # 计算总多项式项数量
            total_terms = np.prod(poly_terms_per_dim)
            
            # 创建设计矩阵
            design_matrix = np.ones((x_segment.shape[0], total_terms))
            
            # 填充设计矩阵（多维多项式基函数）
            col_idx = 0
            indices = [0] * self.n_dim  # 当前多项式幂次
            
            # 递归生成多维多项式基函数
            self._fill_design_matrix_recursive(
                design_matrix, x_norm, indices, 0, col_idx, poly_terms_per_dim
            )
            
            # 使用最小二乘法求解多项式系数
            segment_coeffs, _, _, _ = np.linalg.lstsq(design_matrix, u_segment, rcond=None)
            segment_coeffs = segment_coeffs.reshape(poly_terms_per_dim)
            
        return segment_coeffs
    
    def _fill_design_matrix_recursive(
        self, design_matrix: np.ndarray, x_norm: np.ndarray, 
        indices: List[int], dim: int, col_idx: int, 
        poly_terms_per_dim: List[int]
    ) -> int:
        """递归填充设计矩阵
        
        Args:
            design_matrix: 设计矩阵
            x_norm: 归一化坐标
            indices: 当前多项式幂次
            dim: 当前维度
            col_idx: 当前列索引
            poly_terms_per_dim: 每个维度的多项式项数
            
        Returns:
            next_col_idx: 下一个列索引
        """
        if dim == self.n_dim:
            # 计算当前多项式基函数值
            poly_term = np.ones(x_norm.shape[0])
            for d in range(self.n_dim):
                poly_term *= x_norm[:, d] ** indices[d]
                
            # 存储到设计矩阵的当前列
            design_matrix[:, col_idx] = poly_term
            return col_idx + 1
            
        else:
            next_col_idx = col_idx
            for power in range(poly_terms_per_dim[dim]):
                indices[dim] = power
                next_col_idx = self._fill_design_matrix_recursive(
                    design_matrix, x_norm, indices, dim + 1, 
                    next_col_idx, poly_terms_per_dim
                )
            return next_col_idx
    
    def construct(
        self, data: Dict, model: Any, coeffs: np.ndarray
    ) -> Tuple[np.ndarray, List]:
        """构建解决方案
        
        Args:
            data: 数据字典
            model: 神经网络模型
            coeffs: 拟合系数
            
        Returns:
            predictions: 预测值数组
            segments: 分段信息
        """
        # 获取数据点
        x = data["x"]
        n_points = x.shape[0]
        
        # 初始化预测值数组
        predictions = np.zeros((n_points, 1))
        
        # 计算每个分段的多项式系数数量
        poly_terms_per_dim = [self.poly_degree[i] + 1 for i in range(self.n_dim)]
        coeffs_per_segment = np.prod(poly_terms_per_dim)
        
        # 根据点的位置确定其所在分段，然后应用对应的多项式
        segments = []
        
        for i in range(n_points):
            x_i = x[i]
            
            # 确定点所在的分段
            segment_indices = []
            for d in range(self.n_dim):
                # 在当前维度中查找分段索引
                segment_idx = np.searchsorted(self.segment_ranges[d], x_i[d], side='right') - 1
                # 处理边界情况
                segment_idx = max(0, min(segment_idx, self.config.n_segments[d] - 1))
                segment_indices.append(segment_idx)
            
            # 计算全局分段索引
            global_segment_idx = 0
            for d in range(self.n_dim):
                factor = 1
                for d_inner in range(d):
                    factor *= self.config.n_segments[d_inner]
                global_segment_idx += segment_indices[d] * factor
            
            # 提取当前分段的系数
            start_idx = global_segment_idx * coeffs_per_segment
            end_idx = start_idx + coeffs_per_segment
            segment_coeffs = coeffs[start_idx:end_idx]
            
            # 计算分段边界
            segment_bounds = []
            for d in range(self.n_dim):
                idx = segment_indices[d]
                segment_bounds.append((
                    self.segment_ranges[d][idx],
                    self.segment_ranges[d][idx + 1]
                ))
            
            # 对点使用多项式计算预测值
            predictions[i] = self._eval_poly(x_i, segment_coeffs, segment_bounds)
            
            # 记录点所在的分段
            segments.append(segment_indices)
        
        return predictions, segments
    
    def _eval_poly(
        self, x_i: np.ndarray, coeffs: np.ndarray, 
        segment_bounds: List[Tuple[float, float]]
    ) -> float:
        """使用多项式拟合计算单点的预测值
        
        Args:
            x_i: 单个输入点的坐标 (n_dim,)
            coeffs: 分段的多项式系数
            segment_bounds: 分段边界
            
        Returns:
            pred: 预测值
        """
        # 计算归一化坐标
        x_norm = np.zeros(self.n_dim)
        for d in range(self.n_dim):
            low, high = segment_bounds[d]
            x_norm[d] = 2.0 * (x_i[d] - low) / (high - low) - 1.0
            
        # 计算多项式值（一维情况）
        if self.n_dim == 1:
            poly_val = 0.0
            x_power = 1.0
            
            for j in range(len(coeffs)):
                poly_val += coeffs[j] * x_power
                x_power *= x_norm[0]
                
            return poly_val
            
        else:
            # 多维情况
            poly_terms_per_dim = [self.poly_degree[i] + 1 for i in range(self.n_dim)]
            coeffs = coeffs.reshape(poly_terms_per_dim)
            
            # 递归计算多维多项式值
            return self._eval_poly_recursive(x_norm, coeffs, 0, [])
    
    def _eval_poly_recursive(
        self, x_norm: np.ndarray, coeffs: np.ndarray, 
        dim: int, indices: List[int]
    ) -> float:
        """递归计算多维多项式的值
        
        Args:
            x_norm: 归一化坐标 (n_dim,)
            coeffs: 分段的多项式系数
            dim: 当前维度
            indices: 当前索引列表
            
        Returns:
            poly_val: 多项式值
        """
        if dim == self.n_dim:
            # 根据索引获取系数
            coeff = coeffs[tuple(indices)]
            
            # 计算对应的基函数值
            basis_val = 1.0
            for d in range(self.n_dim):
                basis_val *= x_norm[d] ** indices[d]
                
            return coeff * basis_val
            
        else:
            poly_val = 0.0
            for power in range(self.poly_degree[dim] + 1):
                new_indices = indices + [power]
                poly_val += self._eval_poly_recursive(
                    x_norm, coeffs, dim + 1, new_indices
                )
                
            return poly_val 