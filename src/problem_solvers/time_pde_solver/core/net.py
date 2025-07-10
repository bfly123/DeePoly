import torch
from typing import Dict
from src.abstract_class.base_net import BaseNet

class TimePDENet(BaseNet):
    """时间依赖问题的神经网络实现"""
    
    def physics_loss(self, data_GPU: Dict, data_train: Dict, **kwargs) -> torch.Tensor:
        """计算物理损失
        
        Args:
            data_GPU: GPU数据字典，包含训练所需的GPU数据
            data_train: 训练数据字典，包含训练所需的CPU数据
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
        U_bd = data_GPU["u_bd"]
        param = data_GPU["param"]

        # 获取模型预测
        _, output = self(x_train)

        # 获取参数
        Re = param[0]["Re"]
        nu = param[0]["nu"]

# auto code begin
        # Extract physical quantities from output
        u = output[..., 0]

        # Calculate derivatives in each direction
        du_x = self.gradients(u, x_train)[0][..., 0]

        # Calculate 2nd-order derivatives
        du_xx = self.gradients(du_x, x_train)[0][..., 0]


        # Additional L1 operators (implicit linear)
        L1_extra = [-0.0001*du_xx]

        # Additional L2 operators (semi-implicit linear)
        L2_extra = [u]

        # f_L2 functions (nonlinear functions for L2)
        f_L2 = [u**2-1]

# auto code end

        # 根据不同步骤处理数据
        if step == "1st_order":
            # 一阶方法
            u_n = data_train["u_n"][..., 0]
            # 一阶前向欧拉方法
            eq0 = u - u_n - dt * (L1_0 + L2_0 * f_L2_0)

        elif step == "pre":
            # 第一阶段：Trapezoidal Rule
            u_n = data_train["u_n"][..., 0]
            f_n = data_train["f_n"][..., 0]  # 可能包含右端项

            # u^{n+γ} = u^n + γdt[θF(u^{n+γ}) + (1-θ)F(u^n)]
            eq0 = u - u_n - 0.5 * dt * (L1_0 + L2_0 * f_L2_0 + f_n)

        # 边界条件处理
        _, output_bd = self(x_bd)

        # 应用权重到边界误差
        bc_error = (output_bd[..., 0:2] - U_bd[..., 0:2]) ** 2
        bc_loss = torch.mean(bc_error)

        # 总损失函数
        pde_loss = torch.mean(eq0**2)
        loss = pde_loss + 10 * bc_loss  # 增加边界条件权重

        if torch.rand(1).item() < 0.01:  # 1%的概率打印
            print(f"\nLoss Components ({step} stage):")
            print(f"PDE Loss: {pde_loss.item():.8f}")
            print(f"BC Loss: {bc_loss.item():.8f}")
            print(f"Total Loss: {loss.item():.8f}")

        return loss
    
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