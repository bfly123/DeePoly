import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

class BaseDataGenerator(ABC):
    """基础数据生成器类"""
    
    def __init__(self, config):
        self.config = config
        self.Ns = np.prod(config.n_segments)  # 总段数
        self.Nw = config.points_per_swap  # 边界点数
        self.n_dim = config.n_dim
        self.n_eqs = config.n_eqs

    @abstractmethod
    def generate_global_field(self, x_global: np.ndarray) -> np.ndarray:
        """生成全局初始场"""
        pass

    @abstractmethod
    def generate_data(self, mode: str = "train") -> Dict:
        """生成训练/测试数据"""
        pass
        
    def read_boundary_conditions(self) -> Dict:
        """读取边界条件配置"""
        boundary_dict = {}
        
        # 检查是否存在边界条件配置
        if not hasattr(self.config, 'boundary_conditions') or not self.config.boundary_conditions:
            return boundary_dict
            
        # 初始化每个变量的边界条件字典
        for var in self.config.vars_list:
            boundary_dict[var] = {
                'dirichlet': {
                    'x': [],
                    'u': []
                },
                'neumann': {
                    'x': [],
                    'u': [],
                    'normals': []
                },
                'robin': {
                    'x': [],
                    'u': [],
                    'params': [],
                    'normals': []
                }
            }
        
        # 处理每个边界条件
        for bc in self.config.boundary_conditions:
            region = bc['region']
            bc_type = bc['type'].lower()
            value = bc['value']
            points = bc.get('points', 100)
            
            # 生成边界区域的点
            x_boundary = self._generate_boundary_points(region, points)
            
            # 获取该区域的法向量
            normals = self._get_boundary_normals(region, x_boundary.shape[0])
            
            # 对所有变量应用边界条件
            for var in self.config.vars_list:
                if bc_type == 'dirichlet':
                    # 对于Dirichlet边界条件，直接添加点和值
                    boundary_dict[var]['dirichlet']['x'].append(x_boundary)
                    # 创建对应大小的值数组
                    u_values = np.ones((x_boundary.shape[0], 1)) * value
                    boundary_dict[var]['dirichlet']['u'].append(u_values)
                
                elif bc_type == 'neumann':
                    boundary_dict[var]['neumann']['x'].append(x_boundary)
                    u_values = np.ones((x_boundary.shape[0], 1)) * value
                    boundary_dict[var]['neumann']['u'].append(u_values)
                    boundary_dict[var]['neumann']['normals'].append(normals)
                
                elif bc_type == 'robin':
                    boundary_dict[var]['robin']['x'].append(x_boundary)
                    u_values = np.ones((x_boundary.shape[0], 1)) * value
                    boundary_dict[var]['robin']['u'].append(u_values)
                    boundary_dict[var]['robin']['normals'].append(normals)
                    # 添加Robin边界条件的参数
                    params = bc.get('params', [1.0, 0.0])  # 默认参数
                    boundary_dict[var]['robin']['params'].append(params)
        
        # 合并每个变量的边界条件点和值
        for var in boundary_dict:
            for bc_type in boundary_dict[var]:
                if boundary_dict[var][bc_type]['x']:
                    boundary_dict[var][bc_type]['x'] = np.vstack(boundary_dict[var][bc_type]['x'])
                    boundary_dict[var][bc_type]['u'] = np.vstack(boundary_dict[var][bc_type]['u'])
                    
                    # 如果存在法向量，也需要合并
                    if bc_type in ['neumann', 'robin'] and boundary_dict[var][bc_type]['normals']:
                        boundary_dict[var][bc_type]['normals'] = np.vstack(boundary_dict[var][bc_type]['normals'])
                else:
                    boundary_dict[var][bc_type]['x'] = np.array([])
                    boundary_dict[var][bc_type]['u'] = np.array([])
                    if bc_type in ['neumann', 'robin']:
                        boundary_dict[var][bc_type]['normals'] = np.array([])
        
        return boundary_dict
    
    def _generate_boundary_points(self, region: str, points: int) -> np.ndarray:
        """生成指定边界区域的点"""
        x_domain = self.config.x_domain
        
        if self.n_dim == 1:
            if region == 'left':
                return np.array([[x_domain[0][0]]])
            elif region == 'right':
                return np.array([[x_domain[0][1]]])
        
        elif self.n_dim == 2:
            x_min, x_max = x_domain[0][0], x_domain[0][1]
            y_min, y_max = x_domain[1][0], x_domain[1][1]
            
            if region == 'left':
                y_coords = np.linspace(y_min, y_max, points)
                return np.column_stack((np.ones(points) * x_min, y_coords))
            elif region == 'right':
                y_coords = np.linspace(y_min, y_max, points)
                return np.column_stack((np.ones(points) * x_max, y_coords))
            elif region == 'bottom':
                x_coords = np.linspace(x_min, x_max, points)
                return np.column_stack((x_coords, np.ones(points) * y_min))
            elif region == 'top':
                x_coords = np.linspace(x_min, x_max, points)
                return np.column_stack((x_coords, np.ones(points) * y_max))
        
        elif self.n_dim == 3:
            x_min, x_max = x_domain[0][0], x_domain[0][1]
            y_min, y_max = x_domain[1][0], x_domain[1][1]
            z_min, z_max = x_domain[2][0], x_domain[2][1]
            
            points_per_dim = int(np.sqrt(points))
            
            if region == 'left' or region == 'right':
                x_val = x_min if region == 'left' else x_max
                y_coords = np.linspace(y_min, y_max, points_per_dim)
                z_coords = np.linspace(z_min, z_max, points_per_dim)
                y_grid, z_grid = np.meshgrid(y_coords, z_coords)
                return np.column_stack((
                    np.ones(points_per_dim**2) * x_val,
                    y_grid.flatten(),
                    z_grid.flatten()
                ))
            # 其他3D边界条件可根据需要扩展
            
        return np.array([])

    def _get_boundary_normals(self, region: str, num_points: int) -> np.ndarray:
        """获取边界的法向量"""
        # 初始化法向量为零向量
        normals = np.zeros((num_points, self.n_dim))
        
        if self.n_dim == 1:
            if region == 'left':
                normals[:, 0] = -1.0  # 左边界法向外侧为负向
            elif region == 'right':
                normals[:, 0] = 1.0   # 右边界法向外侧为正向
            
        elif self.n_dim == 2:
            if region == 'left':
                normals[:, 0] = -1.0  # 向左的法向量(-1, 0)
            elif region == 'right':
                normals[:, 0] = 1.0   # 向右的法向量(1, 0)
            elif region == 'bottom':
                normals[:, 1] = -1.0  # 向下的法向量(0, -1)
            elif region == 'top':
                normals[:, 1] = 1.0   # 向上的法向量(0, 1)
            
        elif self.n_dim == 3:
            if region == 'left':
                normals[:, 0] = -1.0  # (-1, 0, 0)
            elif region == 'right':
                normals[:, 0] = 1.0   # (1, 0, 0)
            elif region == 'bottom':
                normals[:, 1] = -1.0  # (0, -1, 0)
            elif region == 'top':
                normals[:, 1] = 1.0   # (0, 1, 0)
            elif region == 'back':
                normals[:, 2] = -1.0  # (0, 0, -1)
            elif region == 'front':
                normals[:, 2] = 1.0   # (0, 0, 1)
        
        return normals

    def split_global_field(self, x_global: np.ndarray, u_global: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """将全局场切分到各个局部段"""
        x_segments = []
        u_segments = []
        
        for n in range(self.Ns):
            mask = self._create_segment_mask(x_global, n)
            x_seg = x_global[mask]
            u_seg = u_global[mask]
            x_segments.append(x_seg)
            u_segments.append(u_seg)
            
        return x_segments, u_segments

    def _generate_global_points(self, mode: str) -> np.ndarray:
        """生成全局采样点"""
        if mode == "train":
            Np_total = self.config.points_domain
            points = []
            for i in range(self.n_dim):
                points.append(np.random.uniform(
                    self.config.x_domain[i, 0],
                    self.config.x_domain[i, 1],
                    Np_total
                ))
            return np.column_stack(points)
        else:
            if isinstance(self.config.points_domain_test, int):
                Np_total = self.config.points_domain_test
                points = []
                for i in range(self.n_dim):
                    points.append(np.random.uniform(
                        self.config.x_domain[i, 0]+0.01,
                        self.config.x_domain[i, 1]-0.01,
                        Np_total
                    ))
                return np.column_stack(points)
            else:
                grids = []
                for i in range(self.n_dim):
                    grids.append(np.linspace(
                        self.config.x_domain[i, 0]+0.01,
                        self.config.x_domain[i, 1]-0.01,
                        self.config.points_domain_test[i]
                    ))
                return np.array(np.meshgrid(*grids)).reshape(self.n_dim, -1).T

    def _create_segment_mask(self, x: np.ndarray, i: int) -> np.ndarray:
        """创建段掩码"""
        x_min, x_max = self.config.x_min, self.config.x_max
        masks = []
        for j in range(self.n_dim):
            masks.append(x[:, j] > x_min[i, j])
            masks.append(x[:, j] <= x_max[i, j])
        
        main_mask = np.logical_and.reduce(masks)
        
        if i == 0:
            boundary_mask = np.logical_or.reduce([
                np.isclose(x[:, j], x_min[i, j]) for j in range(self.n_dim)
            ])
            return np.logical_or(main_mask, boundary_mask)
        return main_mask

    def _normalize_data(self, x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
        """数据归一化"""
        x = np.asarray(x, dtype=np.float64)
        x_min = np.asarray(x_min, dtype=np.float64)
        x_max = np.asarray(x_max, dtype=np.float64)

        normalized = np.zeros_like(x)
        for i in range(x.shape[-1]):
            if x_max[..., i] - x_min[..., i] > 1e-10:
                normalized[..., i] = (x[..., i] - x_min[..., i]) / (
                    x_max[..., i] - x_min[..., i]
                )
            else:
                normalized[..., i] = x[..., i]
        return normalized

    def _process_segments(self, x_segments: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Dict]]:
        """处理各个段的数据"""
        x_segments_norm = []
        x_swap = np.zeros((self.Ns, 2*self.n_dim, self.Nw, self.n_dim), dtype=np.float64)
        x_swap_norm = np.zeros_like(x_swap)
        
        # 获取全局边界条件
        global_boundary_dict = self.read_boundary_conditions()
        
        # 为每个分段创建边界条件字典
        boundary_segments_dict = []
        
        for n in range(self.Ns):
            # 处理分段数据归一化
            x_segments_norm.append(self._normalize_data(x_segments[n], self.config.x_min[n], self.config.x_max[n]))
            
            # 处理交换点
            if self.config.points_per_swap > 0:
                x_swap_n = self._generate_swap_points(n)
                x_swap[n] = x_swap_n
                x_swap_norm[n] = self._normalize_swap_data(x_swap_n, n)
            
            # 处理当前分段的边界条件
            segment_boundary_dict = self._process_segment_boundary(global_boundary_dict, n)
            boundary_segments_dict.append(segment_boundary_dict)
        
        return x_swap, x_swap_norm, x_segments_norm, boundary_segments_dict
    
    def _generate_swap_points(self, n: int) -> np.ndarray:
        """生成交换点"""
        x_min, x_max = self.config.x_min[n], self.config.x_max[n]
        x_swap = np.zeros((2*self.n_dim, self.Nw, self.n_dim))
        
        # 为每个维度生成边界点
        for i in range(self.n_dim):
            # 下边界
            x_swap[2*i] = self._generate_swap_boundary_points(x_min, x_max, i, 0)
            # 上边界
            x_swap[2*i+1] = self._generate_swap_boundary_points(x_min, x_max, i, 1)
            
        return x_swap

    def _generate_swap_boundary_points(self, x_min: np.ndarray, x_max: np.ndarray, dim: int, is_upper: int) -> np.ndarray:
        """生成单个维度的交换边界点"""
        points = np.zeros((self.Nw, self.n_dim))
        points[:, dim] = x_max[dim] if is_upper else x_min[dim]
        
        for i in range(self.n_dim):
            if i != dim:
                points[:, i] = np.linspace(x_min[i], x_max[i], self.Nw)
                
        return points

    def _normalize_swap_data(self, x_swap: np.ndarray, n: int) -> np.ndarray:
        """归一化交换点数据"""
        x_swap_norm = np.zeros_like(x_swap)
        for j in range(x_swap.shape[0]):
            for k in range(x_swap.shape[1]):
                x_swap_norm[j, k] = self._normalize_data(
                    x_swap[j, k], self.config.x_min[n], self.config.x_max[n]
                )
        return x_swap_norm
    
    def _process_segment_boundary(self, global_boundary_dict: Dict, segment_idx: int) -> Dict:
        """处理单个分段的边界条件"""
        # 初始化该分段的边界条件字典
        segment_boundary_dict = {}
        x_min, x_max = self.config.x_min[segment_idx], self.config.x_max[segment_idx]
        
        # 处理每个变量
        for var in global_boundary_dict:
            segment_boundary_dict[var] = {
                'dirichlet': {'x': [], 'u': []},
                'neumann': {'x': [], 'u': [], 'normals': []},
                'robin': {'x': [], 'u': [], 'params': [], 'normals': []}
            }
            
            # 处理不同类型的边界条件
            for bc_type in global_boundary_dict[var]:
                if len(global_boundary_dict[var][bc_type]['x']) == 0:
                    continue
                    
                # 找出在当前分段内的边界点
                x_boundary = global_boundary_dict[var][bc_type]['x']
                mask = self._create_segment_mask(x_boundary, segment_idx)
                
                if not np.any(mask):
                    continue
                    
                # 提取该分段的边界点和值
                x_seg = x_boundary[mask]
                u_seg = global_boundary_dict[var][bc_type]['u'][mask]
                
                # 归一化坐标
                x_seg_norm = self._normalize_data(x_seg, x_min, x_max)
                
                # 保存到该分段的边界条件字典
                segment_boundary_dict[var][bc_type]['x'] = x_seg_norm
                segment_boundary_dict[var][bc_type]['u'] = u_seg
                
                # 处理法向量(如果有)
                if bc_type in ['neumann', 'robin'] and 'normals' in global_boundary_dict[var][bc_type]:
                    normals_seg = global_boundary_dict[var][bc_type]['normals'][mask]
                    segment_boundary_dict[var][bc_type]['normals'] = normals_seg
                    
                # 处理Robin参数(如果有)
                if bc_type == 'robin' and 'params' in global_boundary_dict[var][bc_type]:
                    segment_boundary_dict[var][bc_type]['params'] = global_boundary_dict[var][bc_type]['params']
        
        return segment_boundary_dict

    def _prepare_output_dict(self, *args) -> Dict:
        """准备输出数据字典"""
        [
            x_segments,
            u_segments,
            x_segments_norm,
            x_swap,
            x_swap_norm,
            boundary_segments_dict,
        ] = args

        return {
            "x": np.vstack(x_segments),
            "u": np.vstack(u_segments),
            "x_min": self.config.x_min,
            "x_max": self.config.x_max,
            "x_swap_norm": x_swap_norm,
            "x_swap": x_swap,
            "x_segments_norm": x_segments_norm,
            "x_segments": x_segments,
            "u_segments": u_segments,
            "boundary_segments_dict": boundary_segments_dict,
        }
