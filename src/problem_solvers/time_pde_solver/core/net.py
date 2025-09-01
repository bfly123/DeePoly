import torch
import numpy as np
from typing import Dict
from src.abstract_class.base_net import BaseNet

class TimePDENet(BaseNet):
    """时间依赖问题的神经网络实现"""
    
    def physics_loss(self, data_GPU: Dict, **kwargs) -> torch.Tensor:
        """计算物理损失
        
        Args:
            data_GPU: GPU数据字典，包含训练所需的GPU数据
            **kwargs: 额外参数
                - dt: 时间步长，默认为0
                - step: 时间步类型，默认为"pre"
            
        Returns:
            torch.Tensor: 损失值
        """
        # 获取可选参数
        dt = kwargs.get('dt', 0)
        step = kwargs.get('step', 'pre')
        
        # 获取训练数据
        x_train = data_GPU["x_train"]
        x_bd = data_GPU["x_bd"]
        u_bd = data_GPU["u_bd"]
        param = data_GPU["param"]
        U_n = data_GPU["U_current"]

        # 获取模型预测
        _, U = self(x_train)

        # 获取参数
        Re = param[0]["Re"]
        nu = param[0]["nu"]

# auto code begin
        # Extract physical quantities from output
        u = U[..., 0]

        # Calculate derivatives in each direction
        du_x = self.gradients(u, x_train)[0][..., 0]

        # Calculate 2nd-order derivatives
        du_xx = self.gradients(du_x, x_train)[0][..., 0]

        # L1 operators
        L1 = [0.0001*du_xx]

        # L2 operators
        L2 = [u]

        # F operators
        F = [5-5*u**2]

        # N operators
        N = [
        ]

# auto code end

        # 使用一阶前向欧拉格式：u^{n+1} = u^n + dt * [L1(u^{n+1}) + L2(u^n)*F(u^n)]
        # 对于神经网络训练，我们直接最小化PDE残差
        # PDE残差: du/dt - L1(u) - L2(u)*F(u) = 0
        # 使用一阶差分近似: (u^{n+1} - u^n)/dt - L1(u^{n+1}) - L2(u^n)*F(u^n) = 0
        
        
        pde_loss = 0.0
        dt = 0.01
        
        # 时间演化格式: (u - u_n)/dt = L1[i] + L2[i]*F[i]  
        for i in range(self.config.n_eqs):
            residual_i = (U[:,i] - U_n[:,i])/dt - L1[i] - L2[i]*F[i]
            pde_loss += torch.mean(residual_i**2)


        boundary_loss = 0.0
        boundary_loss_weight = 10.0  # 边界条件权重
        global_boundary_dict = data_GPU.get("global_boundary_dict", None)

        # 处理边界条件 - 纯抽象U处理
        if global_boundary_dict:
            # 处理每个U分量的边界条件
            for var_idx in global_boundary_dict:
                # 处理Dirichlet边界条件
                if (
                    "dirichlet" in global_boundary_dict[var_idx]
                    and global_boundary_dict[var_idx]["dirichlet"]["x"].shape[0] > 0
                ):
                    x_bc = global_boundary_dict[var_idx]["dirichlet"]["x"]
                    u_bc = global_boundary_dict[var_idx]["dirichlet"]["values"]

                    # 在边界点获取模型预测
                    _, pred_bc = self(x_bc)

                    # 计算边界损失 (预测值与目标值之间的MSE)
                    bc_error = (pred_bc - u_bc) ** 2
                    boundary_loss += torch.mean(bc_error)

                # 处理Neumann边界条件
                if (
                    "neumann" in global_boundary_dict[var_idx]
                    and global_boundary_dict[var_idx]["neumann"]["x"].shape[0] > 0
                ):
                    x_bc = global_boundary_dict[var_idx]["neumann"]["x"]
                    u_bc = global_boundary_dict[var_idx]["neumann"]["values"]
                    normals = global_boundary_dict[var_idx]["neumann"]["normals"]

                    # 存储所有法向导数误差
                    all_derivatives_errors = []

                    # 计算每个边界点的法向导数
                    for i in range(x_bc.shape[0]):
                        x_point = x_bc[i : i + 1].clone().detach().requires_grad_(True)

                        # 获取预测
                        _, u_pred = self(x_point)

                        # 计算梯度
                        grads = self.gradients(u_pred, x_point)[0]

                        # 计算法向导数 (梯度与法向量的点积)
                        normal = normals[i]
                        normal_derivative = torch.sum(grads * normal)

                        # 计算误差
                        bc_error = (normal_derivative - u_bc[i]) ** 2
                        all_derivatives_errors.append(bc_error)

                    # 计算所有Neumann边界点的MSE
                    if all_derivatives_errors:
                        neumann_errors = torch.stack(all_derivatives_errors)
                        boundary_loss += torch.mean(neumann_errors)

                # 处理Robin边界条件
                if (
                    "robin" in global_boundary_dict[var_idx]
                    and global_boundary_dict[var_idx]["robin"]["x"].shape[0] > 0
                ):
                    x_bc = global_boundary_dict[var_idx]["robin"]["x"]
                    u_bc = global_boundary_dict[var_idx]["robin"]["values"]
                    normals = global_boundary_dict[var_idx]["robin"]["normals"]
                    params = global_boundary_dict[var_idx]["robin"]["params"]

                    # 存储所有Robin边界条件误差
                    all_robin_errors = []

                    # 处理每个Robin边界点
                    for i in range(x_bc.shape[0]):
                        x_point = x_bc[i : i + 1].clone().detach().requires_grad_(True)

                        # 获取预测
                        _, u_pred = self(x_point)

                        # 获取参数
                        alpha, beta = params[0], params[1]

                        # 如果beta非零则计算梯度
                        if abs(beta) > 1e-10:
                            grads = self.gradients(u_pred, x_point)[0]

                            # 计算法向导数
                            normal = normals[i]
                            normal_derivative = torch.sum(grads * normal)

                            # Robin条件: alpha*u + beta*du/dn = g
                            robin_value = alpha * u_pred + beta * normal_derivative
                            bc_error = (robin_value - u_bc[i]) ** 2
                        else:
                            # 如果beta为零，则相当于Dirichlet条件
                            bc_error = (alpha * u_pred - u_bc[i]) ** 2
                        
                        all_robin_errors.append(bc_error)
                    
                    # 计算所有Robin边界点的MSE
                    if all_robin_errors:
                        robin_errors = torch.stack(all_robin_errors)
                        boundary_loss += torch.mean(robin_errors)
                
                # 处理周期边界条件
                if 'periodic' in global_boundary_dict[var_idx] and global_boundary_dict[var_idx]['periodic']['pairs']:
                    for pair in global_boundary_dict[var_idx]['periodic']['pairs']:
                        x_bc_1 = pair['x_1']
                        x_bc_2 = pair['x_2']
                        constraint_type = pair['constraint_type']
                        
                        # 在两组边界点获取模型预测
                        _, pred_bc_1 = self(x_bc_1)
                        _, pred_bc_2 = self(x_bc_2)
                        
                        if constraint_type == 'dirichlet':
                            # 周期Dirichlet: U(x1) = U(x2)
                            periodic_error = (pred_bc_1 - pred_bc_2) ** 2
                            boundary_loss += torch.mean(periodic_error)
                        elif constraint_type == 'neumann':
                            # 周期Neumann: ∂U/∂n(x1) = ∂U/∂n(x2)
                            normals_1 = pair['normals_1']
                            normals_2 = pair['normals_2']
                            
                            # 计算两组边界点的法向导数
                            all_periodic_errors = []
                            for i in range(x_bc_1.shape[0]):
                                x_point_1 = x_bc_1[i:i+1].clone().detach().requires_grad_(True)
                                x_point_2 = x_bc_2[i:i+1].clone().detach().requires_grad_(True)
                                
                                _, u_pred_1 = self(x_point_1)
                                _, u_pred_2 = self(x_point_2)
                                
                                grads_1 = self.gradients(u_pred_1, x_point_1)[0]
                                grads_2 = self.gradients(u_pred_2, x_point_2)[0]
                                
                                normal_1 = normals_1[i]
                                normal_2 = normals_2[i]
                                
                                normal_deriv_1 = torch.sum(grads_1 * normal_1)
                                normal_deriv_2 = torch.sum(grads_2 * normal_2)
                                
                                periodic_error = (normal_deriv_1 - normal_deriv_2) ** 2
                                all_periodic_errors.append(periodic_error)
                            
                            if all_periodic_errors:
                                periodic_errors = torch.stack(all_periodic_errors)
                                boundary_loss += torch.mean(periodic_errors)
        
        # 总损失函数
        total_loss = pde_loss + boundary_loss_weight * boundary_loss

        return total_loss
    
    def prepare_gpu_data(self, data: Dict, U_current: np.ndarray = None) -> Dict:
        """Prepare GPU data for time PDE problems
        
        Args:
            data: Input data dictionary containing training data
            
        Returns:
            gpu_data: Dictionary containing GPU tensors
        """
        gpu_data = {}
        
        # Transfer coordinate data to GPU
        gpu_data["x_train"] = torch.tensor(
            data["x"], dtype=torch.float64, device=self.config.device, requires_grad=True
        )
        gpu_data["U_current"] = torch.tensor(
            U_current, dtype=torch.float64, device=self.config.device
        )
        
        # Transfer boundary data to GPU - 纯抽象U处理
        if "global_boundary_dict" in data:
            global_boundary_dict = {}
            for var_idx in data["global_boundary_dict"]:
                global_boundary_dict[var_idx] = {}
                for bc_type in data["global_boundary_dict"][var_idx]:
                    global_boundary_dict[var_idx][bc_type] = {}
                    
                    if bc_type == 'periodic':
                        # 处理周期边界条件的pairs
                        global_boundary_dict[var_idx][bc_type]['pairs'] = []
                        for pair in data["global_boundary_dict"][var_idx][bc_type]['pairs']:
                            gpu_pair = {}
                            for pair_key, pair_value in pair.items():
                                if isinstance(pair_value, np.ndarray) and pair_value.size > 0:
                                    if 'x_' in pair_key:  # x_1, x_2
                                        gpu_pair[pair_key] = torch.tensor(
                                            pair_value, dtype=torch.float64, device=self.config.device, requires_grad=True
                                        )
                                    else:  # normals_1, normals_2等
                                        gpu_pair[pair_key] = torch.tensor(
                                            pair_value, dtype=torch.float64, device=self.config.device
                                        )
                                else:
                                    gpu_pair[pair_key] = pair_value
                            global_boundary_dict[var_idx][bc_type]['pairs'].append(gpu_pair)
                    else:
                        # 处理常规边界条件
                        for key, value in data["global_boundary_dict"][var_idx][bc_type].items():
                            if isinstance(value, np.ndarray) and value.size > 0:
                                if key == "x":
                                    global_boundary_dict[var_idx][bc_type][key] = torch.tensor(
                                        value, dtype=torch.float64, device=self.config.device, requires_grad=True
                                    )
                                else:
                                    global_boundary_dict[var_idx][bc_type][key] = torch.tensor(
                                        value, dtype=torch.float64, device=self.config.device
                                    )
                            else:
                                global_boundary_dict[var_idx][bc_type][key] = value
            gpu_data["global_boundary_dict"] = global_boundary_dict
        else:
            gpu_data["global_boundary_dict"] = {}
        
        # Transfer boundary points and values for physics loss computation
        if "global_boundary_dict" in data and data["global_boundary_dict"]:
            # Extract boundary data for easier access in physics_loss - 纯抽象U处理
            x_bd_list = []
            u_bd_list = []
            for var_idx in data["global_boundary_dict"]:
                for bc_type in data["global_boundary_dict"][var_idx]:
                    if "x" in data["global_boundary_dict"][var_idx][bc_type]:
                        x_bd = data["global_boundary_dict"][var_idx][bc_type]["x"]
                        u_bd = data["global_boundary_dict"][var_idx][bc_type]["values"]
                        if isinstance(x_bd, np.ndarray) and x_bd.size > 0:
                            x_bd_list.append(x_bd)
                            u_bd_list.append(u_bd)
            
            if x_bd_list:
                gpu_data["x_bd"] = torch.tensor(
                    np.vstack(x_bd_list), dtype=torch.float64, device=self.config.device, requires_grad=True
                )
                gpu_data["u_bd"] = torch.tensor(
                    np.vstack(u_bd_list), dtype=torch.float64, device=self.config.device
                )
            else:
                # Create empty tensors if no boundary data
                gpu_data["x_bd"] = torch.zeros((0, self.config.n_dim), dtype=torch.float64, device=self.config.device)
                gpu_data["u_bd"] = torch.zeros((0, self.config.n_eqs), dtype=torch.float64, device=self.config.device)
        else:
            # Create empty tensors if no boundary data
            gpu_data["x_bd"] = torch.zeros((0, self.config.n_dim), dtype=torch.float64, device=self.config.device)
            gpu_data["u_bd"] = torch.zeros((0, self.config.n_eqs), dtype=torch.float64, device=self.config.device)
        
        # Add parameter data for physics loss
        gpu_data["param"] = [{"Re": 1.0, "nu": 1.0}]  # Default parameters for time PDE
        
        return gpu_data
    
    @staticmethod
    def model_init(config):
        """初始化模型
        
        Args:
            config: 配置对象
            
        Returns:
            TimePDENet: 初始化后的模型
        """
        model = TimePDENet(
            in_dim=config.n_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.n_eqs
        ).to(config.device)
        return model 