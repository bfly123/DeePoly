from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from torch import nn

# Delay import to break cycle
# from .features_generator import FeatureGenerator


class BaseDeepPolyFitter(ABC):
    """拟合器抽象基类"""

    def __init__(self, config, dt: float = None, data: Dict = None):
        # Import FeatureGenerator here
        from .features_generator import FeatureGenerator
        self.data = data


        self.config = config
        self.feature_generator = FeatureGenerator(self.config.linear_device)
        self.dt = dt
        # Ensure all_derivatives exists in config
       # self.all_derivatives = getattr(
       #     config, "all_derivatives", []
       # )  # Provide default if missing
        self.all_derivatives = config.all_derivatives
        self.ns = np.prod(self.config.n_segments)
        self.n_segments = self.config.n_segments
        self.dg = np.int32(
            np.prod(
                self.config.poly_degree + np.ones(np.shape(self.config.poly_degree))
            )
        )
        if self.config.method == "hybrid":
            self.dgN = self.config.DNN_degree + self.dg  # Use getattr
        else:
            self.dgN = self.dg
        self.device = self.config.linear_device
        self.n_eqs = self.config.n_eqs

        # Initialize members
        self.A = None
        self.b = None
        self.equations = None
        self.variables = None

    def get_matrix_size(
        self, ns: int, ne: int, dgN: int,
    ) -> Tuple[int, int, List[int]]:
        """计算矩阵大小，支持任意维度"""
        total_rows = 0
        rows_per_segment = []
        for i in range(ns):
            n_points = len(self.data["x_segments"][i])
            segment_rows = n_points * ne
            rows_per_segment.append(segment_rows)
            total_rows += segment_rows

        n_cols = ne * dgN * ns
        return total_rows, n_cols, rows_per_segment

#    @abstractmethod
#    def get_segment_data(self, segment_idx: int) -> Dict:
#        """获取指定段的数据，支持高维索引"""
#        pass

    def process_constraints(
        self, A: List[np.ndarray], b: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """处理约束条件"""
        A_constraints = np.vstack(A)
        b_constraints = np.vstack([b_i.reshape(-1, 1) for b_i in b])
        return A_constraints, b_constraints

    @abstractmethod
    def _build_segment_jacobian(
        self,
        segment_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构建单个段的雅可比矩阵"""
        pass

    def fitter_init(self, model: nn.Module):
        """初始化拟合器并构建系统方程"""
        model = model.to(self.config.linear_device)
        self._build_system(model)

    def fit(self, **kwargs) -> np.ndarray:
        """使用非线性求解器求解系统"""
        coeffs = self._solve_system(**kwargs)
        return coeffs

    def _solve_system(self, **kwargs) -> np.ndarray:
        """使用信赖域法求解非线性系统"""
        shape = (self.ns, self.config.n_eqs, self.dgN)
        L, r = self._build_jacobian(**kwargs)
        coeffs = self.solver.solve(L, r, method="auto")[0]
        return coeffs.reshape(shape)

    def _build_jacobian(
        self, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:

        ns = self.ns
        ne = self.config.n_eqs
        dgN = self.dgN

        # 计算总行数
        total_rows = 0
        rows_per_segment = []
        for i in range(ns):
            n_points = len(self.data["x_segments_norm"][i])
            segment_rows = n_points * ne
            rows_per_segment.append(segment_rows)
            total_rows += segment_rows

#TODO: dgN may can change in future
        n_cols = ne * dgN * ns

        # 预分配矩阵
        J1 = np.zeros((total_rows, n_cols), dtype=np.float64)
        b1 = np.zeros((total_rows, 1), dtype=np.float64)

        # 处理每个段
        row_start = 0
        for i in range(ns):
            # 计算当前段的索引
            row_end = row_start + rows_per_segment[i]
            col_start = i * ne * dgN
            col_end = (i + 1) * ne * dgN

            L, r = self._build_segment_jacobian(i)

            # 填充矩阵
            J1[row_start:row_end, col_start:col_end] = L
            b1[row_start:row_end] = r

            row_start = row_end

        # 处理约束
        if len(self.A) > 0:
            A_constraints = np.vstack(self.A)
            b_constraints = np.vstack([b_i.reshape(-1, 1) for b_i in self.b])
        else:
            A_constraints = np.zeros((0, n_cols))
            b_constraints = np.zeros((0, 1))

        # 组装最终结果
        J = np.vstack([J1, A_constraints])
        r = np.vstack([b1, b_constraints])

        return J, r


    def _get_segment_features(
        self,
        x: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
        model: nn.Module,
        derivative: List[int] = None,
    ) -> np.ndarray:
        """获取段特征"""
        n_dims = x.shape[1]
        if derivative is None:
            derivative = [0] * n_dims

        # FeatureGenerator is already initialized in self.feature_generator
        X_poly = self.feature_generator.sniper_features(
            x, self.config.poly_degree, derivative
        )

        scale_factor = 1.0
        for dim, der in enumerate(derivative):
            if der > 0 and (x_max[dim] - x_min[dim]) > 1e-9:  # Avoid division by zero
                scale_factor /= (x_max[dim] - x_min[dim]) ** der
            elif der > 0:
                # Handle zero range case if necessary, e.g., warning or error
                print(
                    f"Warning: Zero range encountered for dimension {dim} during scaling."
                )
                scale_factor = 0  # Or handle appropriately
                break
        X_poly *= scale_factor

        if getattr(self.config, "method", "hybrid") == "hybrid":
            X_nn = self.feature_generator.spotter_features(
                x, x_min, x_max, model, derivative, self.device
            )
            X_poly = np.hstack([X_poly, X_nn])

        return X_poly

    def _build_system(self, model: nn.Module):
        """构建系统方程和约束"""
        self.equations = {f"eq{i}": [] for i in range(self.config.n_eqs)}
        self.variables = {}

        for term in self.config.eq_nonlinear_list:
            name = term[2]
            if name not in self.variables:
                self.variables[name] = []

        self._build_basic_equations_and_varibales(model)
        self._add_constraints(model)

    def _build_basic_equations_and_varibales(self, model: nn.Module):
        """构建基本方程组"""
        for i in range(self.ns):
            features = self._get_features(i, model)
            self._build_segment_equations_and_variables(features)

    def _get_features(self, segment_idx: int, model: nn.Module) -> List[np.ndarray]:
        """获取特征和导数"""
        feature_list = self.config.deriv_orders
        x = self.data["x_segments_norm"][segment_idx]
        x_min = self.config.x_min[segment_idx]
        x_max = self.config.x_max[segment_idx]

        features = []
        for derivative in feature_list:
            features.append(
                self._get_segment_features(x, x_min, x_max, model, derivative)
            )
        return features

    def _build_segment_equations_and_variables(self, features: List[np.ndarray]):
        """构建段的方程和变量"""
        ne = self.config.n_eqs
        n_points = features[0].shape[0]

        eq_list = self.config.eq_linear_list
        nonlinear_eq_list = self.config.eq_nonlinear_list

        for i in range(ne):
            eq = np.zeros((n_points, ne * self.dgN))
            for term in eq_list[i]:
                coeff, var, deriv = term[0], term[1], term[2]
                j = var * self.dgN
                eq[:, j : j + self.dgN] += coeff * features[deriv]
            self.equations[f"eq{i}"].append(eq)

        for term in nonlinear_eq_list:
            var, deriv, name = term[0], term[1], term[2]
            U0 = np.zeros((n_points, ne * self.dgN))
            j = var * self.dgN
            U0[:, j : j + self.dgN] = features[deriv]
            self.variables[name].append(U0)

    def _add_constraints(self, model: nn.Module):
        """添加约束条件"""
        self.A, self.b = [], []
        if self.config.problem_type != "func_fitting":
            self._add_boundary_conditions(model)
            self._add_swap_conditions(model)

    def _add_boundary_conditions(self, model: nn.Module):
        """添加边界条件"""
        ne = self.config.n_eqs
        
        # 遍历所有分段
        for i in range(self.ns):
            # 获取当前分段的边界条件数据
            if 'boundary_segments_dict' not in self.data or i >= len(self.data['boundary_segments_dict']):
                continue
            
            segment_boundaries = self.data['boundary_segments_dict'][i]
            
            # 处理各个变量的边界条件
            for var_idx, var in enumerate(self.config.vars_list):
                if var not in segment_boundaries:
                    continue
                
                # 处理Dirichlet边界条件
                if 'dirichlet' in segment_boundaries[var] and len(segment_boundaries[var]['dirichlet']['x']) > 0:
                    x_bd = segment_boundaries[var]['dirichlet']['x']
                    u_bd = segment_boundaries[var]['dirichlet']['u']
                    
                    # 获取特征
                    features = self._get_segment_features(
                        x_bd, self.config.x_min[i], self.config.x_max[i], model, [0] * self.config.n_dim
                    )
                    
                    # 构建约束矩阵
                    constraint = np.zeros((x_bd.shape[0], self.dgN * self.ns * ne))
                    start_idx = i * self.dgN * ne + self.dgN * var_idx
                    constraint[:, start_idx : start_idx + self.dgN] = features
                    
                    # 添加约束
                    self.A.append(10 * constraint)
                    self.b.append(10 * u_bd.flatten())
                
                # 处理Neumann边界条件
                if 'neumann' in segment_boundaries[var] and len(segment_boundaries[var]['neumann']['x']) > 0:
                    x_bd = segment_boundaries[var]['neumann']['x']
                    u_bd = segment_boundaries[var]['neumann']['u']
                    normals = segment_boundaries[var]['neumann']['normals']
                    
                    # 对每个边界点处理
                    for pt_idx in range(x_bd.shape[0]):
                        point = x_bd[pt_idx:pt_idx+1]  # 保持维度
                        normal = normals[pt_idx]
                        
                        # 对每个空间维度，计算法向导数
                        for dim in range(self.config.n_dim):
                            if abs(normal[dim]) < 1e-10:  # 如果法向量在该方向分量很小，跳过
                                continue
                            
                            # 创建导数向量
                            derivative = [0] * self.config.n_dim
                            derivative[dim] = 1  # 一阶导数
                            
                            # 获取该方向导数的特征
                            features = self._get_segment_features(
                                point, self.config.x_min[i], self.config.x_max[i], model, derivative
                            )
                            
                            # 构建约束矩阵
                            constraint = np.zeros((1, self.dgN * self.ns * ne))
                            start_idx = i * self.dgN * ne + self.dgN * var_idx
                            constraint[:, start_idx : start_idx + self.dgN] = normal[dim] * features
                            
                            # 添加约束
                            self.A.append(10 * constraint)
                            self.b.append(10 * np.array([u_bd[pt_idx, 0]]))
                
                # Robin边界条件处理类似，需要同时考虑值和导数
                if 'robin' in segment_boundaries[var] and len(segment_boundaries[var]['robin']['x']) > 0:
                    x_bd = segment_boundaries[var]['robin']['x']
                    u_bd = segment_boundaries[var]['robin']['u']
                    normals = segment_boundaries[var]['robin']['normals']
                    params = segment_boundaries[var]['robin']['params']  # 通常是[alpha, beta]形式
                    
                    # Robin边界条件形式: alpha*u + beta*du/dn = g
                    # 这里u_bd包含g值
                    if not isinstance(params, list) or len(params) < 2:
                        params = [1.0, 0.0]  # 默认参数
                    
                    alpha, beta = params
                    
                    # 处理每个边界点
                    for pt_idx in range(x_bd.shape[0]):
                        point = x_bd[pt_idx:pt_idx+1]
                        normal = normals[pt_idx]
                        
                        # 获取u的特征
                        features_u = self._get_segment_features(
                            point, self.config.x_min[i], self.config.x_max[i], model, [0] * self.config.n_dim
                        )
                        
                        # 初始化法向导数特征
                        features_du_dn = np.zeros_like(features_u)
                        
                        # 计算法向导数
                        for dim in range(self.config.n_dim):
                            if abs(normal[dim]) < 1e-10:
                                continue
                            
                            derivative = [0] * self.config.n_dim
                            derivative[dim] = 1
                            
                            features_dim = self._get_segment_features(
                                point, self.config.x_min[i], self.config.x_max[i], model, derivative
                            )
                            
                            features_du_dn += normal[dim] * features_dim
                        
                        # 构建约束矩阵
                        constraint = np.zeros((1, self.dgN * self.ns * ne))
                        start_idx = i * self.dgN * ne + self.dgN * var_idx
                        constraint[:, start_idx : start_idx + self.dgN] = alpha * features_u + beta * features_du_dn
                        
                        # 添加约束
                        self.A.append(10 * constraint)
                        self.b.append(10 * np.array([u_bd[pt_idx, 0]]))

    def _add_swap_conditions(self, model: nn.Module):
        """Add interface continuity conditions"""
        n_dims = len(self.n_segments)
        
        for dim in range(n_dims):
            current_segments = list(self.n_segments)
            current_segments[dim] -= 1
            
            indices = np.array(
                np.meshgrid(*[range(n) for n in current_segments])
            ).T.reshape(-1, n_dims)

            for idx in indices:
                self._add_dimension_swap(idx, dim, model)

    def _add_dimension_swap(self, idx: np.ndarray, dim: int, model: nn.Module):
        """Add interface continuity conditions for specified dimension"""
        n_dims = len(self.n_segments)
        NS = np.prod(self.n_segments)
        nw = self.config.points_per_swap
        dgN = self.dgN
        ne = self.config.n_eqs

        idx1 = idx.copy()
        idx2 = idx.copy()
        idx2[dim] += 1

        # Use Fortran-style indexing to match base_config.py
        seg1 = np.ravel_multi_index(idx1, self.n_segments, order='F')
        seg2 = np.ravel_multi_index(idx2, self.n_segments, order='F')

        for i in range(ne):
            derivatives = self.all_derivatives[i]
            for der in derivatives:
                self._add_swap_derivative(model, seg1, seg2, NS, nw, dgN, i, der, dim)

    def _add_swap_derivative(
        self,
        model: nn.Module,
        seg1: int,
        seg2: int,
        NS: int,
        nw: int,
        dgN: int,
        eq_idx: int,
        derivative: List[int],
        dim: int,
    ):
        """Add derivative interface conditions for specified dimension"""
        P1 = self._get_segment_features(
            self.data["x_swap_norm"][seg1, 2 * dim + 1, :, :],
            self.config.x_min[seg1],
            self.config.x_max[seg1],
            model,
            derivative,
        )

        P2 = self._get_segment_features(
            self.data["x_swap_norm"][seg2, 2 * dim, :, :],
            self.config.x_min[seg2],
            self.config.x_max[seg2],
            model,
            derivative,
        )

        cont = np.zeros((nw, dgN * NS * self.config.n_eqs), dtype=np.float64)

        ndisp = seg1 * dgN * self.config.n_eqs + dgN * eq_idx
        cont[:, ndisp : ndisp + dgN] = P1

        ndisp = seg2 * dgN * self.config.n_eqs + dgN * eq_idx
        cont[:, ndisp : ndisp + dgN] = -P2

        self.A.extend([cont])
        self.b.extend(np.zeros((nw, 1)).flatten())

    def construct(
        self,
        data: Dict,
        model: nn.Module,
        coeffs: np.ndarray,
        derivative: List[int] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """使用系数进行预测"""
        x_segments = data["x_segments_norm"]
        x_min = self.config.x_min
        x_max = self.config.x_max
        ne = self.config.n_eqs
        ns = self.ns

        if derivative is None:
            derivative = [0] * self.config.n_dim

        total_size = sum(seg.shape[0] for seg in x_segments)
        u_pred = np.zeros((total_size, ne))
        pred_segments = []

        start_idx = 0
        for i in range(ns):
            segment_size = x_segments[i].shape[0]
            end_idx = start_idx + segment_size

            features = self._get_segment_features(
                x_segments[i], x_min[i], x_max[i], model, derivative
            )

            segment_pred = np.zeros((segment_size, ne))
            for j in range(ne):
                segment_pred[:, j] = features @ coeffs[i, j, :]

            u_pred[start_idx:end_idx, :] = segment_pred
            pred_segments.append(segment_pred)
            start_idx = end_idx

        return u_pred, pred_segments

    def predict_segment(
        self, data: Dict, model: nn.Module
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """直接使用神经网络模型预测"""
        model.to(self.config.linear_device)
        x_segments = data["x_segments"]
        ne = self.config.n_eqs
        ns = self.ns

        total_size = sum(seg.shape[0] for seg in x_segments)
        u_pred = np.zeros((total_size, ne))
        pred_segments = []

        start_idx = 0
        for i in range(ns):
            segment_size = x_segments[i].shape[0]
            end_idx = start_idx + segment_size

            x_in = torch.tensor(
                x_segments[i], dtype=torch.float64, device=self.config.linear_device
            )
            _, pred = model(x_in)
            pred = pred.cpu().detach().numpy()

            u_pred[start_idx:end_idx, :] = pred
            pred_segments.append(np.array(pred))

            start_idx = end_idx

        return u_pred, pred_segments
