import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class BoundaryConstraint:
    """纯抽象的边界条件约束表示，只使用变量索引"""
    var_idx: int  # 对应U向量中的分量索引
    constraint_type: str  # 'dirichlet', 'neumann', 'robin', 'periodic'
    x_coords: torch.Tensor  # 边界点坐标
    target_values: Optional[torch.Tensor] = None  # 约束目标值
    normals: Optional[torch.Tensor] = None  # 法向量(Neumann/Robin需要)
    # 周期边界条件特有字段
    x_coords_pair: Optional[torch.Tensor] = None  # 配对边界坐标点
    normals_pair: Optional[torch.Tensor] = None  # 配对边界法向量
    periodic_type: Optional[str] = None  # 周期约束类型：'dirichlet'或'neumann'
    
    def evaluate_dirichlet(self, U_pred: torch.Tensor) -> torch.Tensor:
        """Dirichlet: U[var_idx] = target_values"""
        return U_pred[:, self.var_idx:self.var_idx+1] - self.target_values
    
    def evaluate_neumann(self, U_pred: torch.Tensor, gradients_func) -> torch.Tensor:
        """Neumann: ∂U[var_idx]/∂n = target_values"""
        grads = gradients_func(U_pred[:, self.var_idx:self.var_idx+1], self.x_coords)[0]
        normal_derivative = torch.sum(grads * self.normals, dim=1, keepdim=True)
        return normal_derivative - self.target_values
    
    def evaluate_periodic(self, U_pred_1: torch.Tensor, U_pred_2: torch.Tensor, gradients_func=None) -> torch.Tensor:
        """周期边界条件: 边界对应点的值或导数相等"""
        if self.periodic_type == 'dirichlet':
            # 周期Dirichlet: U(x1) = U(x2)
            return U_pred_1[:, self.var_idx:self.var_idx+1] - U_pred_2[:, self.var_idx:self.var_idx+1]
        elif self.periodic_type == 'neumann':
            # 周期Neumann: ∂U/∂n(x1) = ∂U/∂n(x2)
            grads_1 = gradients_func(U_pred_1[:, self.var_idx:self.var_idx+1], self.x_coords)[0]
            grads_2 = gradients_func(U_pred_2[:, self.var_idx:self.var_idx+1], self.x_coords_pair)[0]
            normal_deriv_1 = torch.sum(grads_1 * self.normals, dim=1, keepdim=True)
            normal_deriv_2 = torch.sum(grads_2 * self.normals_pair, dim=1, keepdim=True)
            return normal_deriv_1 - normal_deriv_2
        else:
            raise ValueError(f"Unknown periodic type: {self.periodic_type}")
    
    def evaluate(self, U_pred: torch.Tensor, gradients_func=None, U_pred_pair: torch.Tensor = None) -> torch.Tensor:
        """统一评估接口"""
        if self.constraint_type == 'dirichlet':
            return self.evaluate_dirichlet(U_pred)
        elif self.constraint_type == 'neumann':
            return self.evaluate_neumann(U_pred, gradients_func)
        elif self.constraint_type == 'periodic':
            if U_pred_pair is None:
                raise ValueError("Periodic constraints require U_pred_pair")
            return self.evaluate_periodic(U_pred, U_pred_pair, gradients_func)
        else:
            raise NotImplementedError(f"Constraint type {self.constraint_type} not implemented")

class BoundaryConstraintManager:
    """边界条件约束管理器 - 纯抽象U处理"""
    
    def __init__(self, config):
        self.config = config
        self.constraints: List[BoundaryConstraint] = []
    
    def build_constraints_from_data(self, boundary_data: Dict) -> None:
        """从抽象边界条件数据构建约束列表 - 纯U向量索引处理"""
        self.constraints.clear()
        
        for var_idx in boundary_data:
            var_boundary_data = boundary_data[var_idx]
            
            # Dirichlet约束: U[var_idx] = target_values
            if self._has_valid_boundary(var_boundary_data, 'dirichlet'):
                constraint = BoundaryConstraint(
                    var_idx=var_idx,
                    constraint_type='dirichlet',
                    x_coords=self._to_tensor(var_boundary_data['dirichlet']['x']),
                    target_values=self._to_tensor(var_boundary_data['dirichlet']['values'])
                )
                self.constraints.append(constraint)
            
            # Neumann约束: ∂U[var_idx]/∂n = target_values
            if self._has_valid_boundary(var_boundary_data, 'neumann'):
                constraint = BoundaryConstraint(
                    var_idx=var_idx,
                    constraint_type='neumann',
                    x_coords=self._to_tensor(var_boundary_data['neumann']['x']),
                    target_values=self._to_tensor(var_boundary_data['neumann']['values']),
                    normals=self._to_tensor(var_boundary_data['neumann']['normals'])
                )
                self.constraints.append(constraint)
            
            # 周期边界约束
            if 'periodic' in var_boundary_data and var_boundary_data['periodic']['pairs']:
                for pair in var_boundary_data['periodic']['pairs']:
                    constraint = BoundaryConstraint(
                        var_idx=var_idx,
                        constraint_type='periodic',
                        x_coords=self._to_tensor(pair['x_1']),
                        x_coords_pair=self._to_tensor(pair['x_2']),
                        periodic_type=pair['constraint_type']
                    )
                    
                    if pair['constraint_type'] == 'neumann':
                        constraint.normals = self._to_tensor(pair['normals_1'])
                        constraint.normals_pair = self._to_tensor(pair['normals_2'])
                    
                    self.constraints.append(constraint)
    
    def compute_boundary_loss(self, U_pred_func, gradients_func, weight: float = 10.0) -> torch.Tensor:
        """计算边界条件损失 - 纯抽象U处理
        
        Args:
            U_pred_func: 函数，输入x_coords返回U_pred
            gradients_func: 梯度计算函数
            weight: 损失权重
        """
        if not self.constraints:
            # 创建一个dummy tensor用于设备检测
            dummy_x = torch.zeros((1, 1))
            device = next(iter(self.config.__dict__.values())) if hasattr(self.config, 'device') else 'cpu'
            return torch.tensor(0.0, device=device)
        
        total_loss = torch.tensor(0.0, device=self.constraints[0].x_coords.device)
        
        for constraint in self.constraints:
            if constraint.constraint_type == 'periodic':
                # 周期边界条件需要两组预测值
                U_pred_1 = U_pred_func(constraint.x_coords)
                U_pred_2 = U_pred_func(constraint.x_coords_pair)
                residual = constraint.evaluate(U_pred_1, gradients_func, U_pred_2)
            else:
                # 常规边界条件
                U_pred = U_pred_func(constraint.x_coords)
                residual = constraint.evaluate(U_pred, gradients_func)
            
            loss = torch.mean(residual ** 2)
            total_loss += loss
        
        return weight * total_loss
    
    def _has_valid_boundary(self, var_boundary_data: Dict, bc_type: str) -> bool:
        """检查是否有有效的边界条件 - 纯抽象处理"""
        return (bc_type in var_boundary_data and 
                isinstance(var_boundary_data[bc_type]['x'], np.ndarray) and 
                var_boundary_data[bc_type]['x'].size > 0)
    
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """将numpy数组转换为tensor"""
        return torch.tensor(data, dtype=torch.float64, device=self.config.device, requires_grad=True)