"""
Spotter Net: 启动步神经网络（复用 BaseNet）
用于直接求解稳态PDE残差 R(U) = 0，区别于后续的伪时间步推进

关键改进：
1. 复用 BaseNet 网络结构，便于参数迁移到 Fitter 阶段
2. 随机采样训练点（每epoch重采样）
3. 自适应学习率衰减
4. float64精度
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from src.abstract_class.base_net import BaseNet
from src.abstract_class.boundary_conditions import BoundaryConditionMixin


class SpotterNet(BoundaryConditionMixin, BaseNet):
    """
    Spotter阶段专用网络 - 继承 BaseNet，便于网络迁移

    与伪时间步推进的区别：
    - Spotter: 求解 R(U) = L1 + L2*F + N - S = 0 (稳态残差)
    - 伪时间步: 求解 ∂U/∂τ + R(U) = 0 (时间演化)
    """

    def __init__(self, config, use_periodic: bool = False):
        """
        Args:
            config: 配置对象
            use_periodic: 是否使用周期性特征（默认False，与原实现保持一致）
        """
        # 调用 BaseNet 初始化（会自动构建网络）
        super().__init__(config, use_periodic=use_periodic)

        # 保存配置供后续使用
        self._n_pde_points = None

    def forward(self, x: torch.Tensor):
        """
        前向传播 - 兼容 BaseNet 的返回格式

        Args:
            x: 输入坐标 [N, n_dim]

        Returns:
            Tuple[h, U]: (隐藏层输出, 预测解) - 与 BaseNet 保持一致
        """
        # 调用父类的 forward 方法（返回 h, U）
        h, U = super().forward(x)
        return h, U

    def physics_loss(self, data_GPU: Dict, **kwargs) -> torch.Tensor:
        """
        计算物理损失 - 与 GeneralPDENet 统一的接口

        Args:
            data_GPU: GPU数据字典
            **kwargs: 额外参数
                - resample: 是否重采样PDE点
                - dt: 伪时间步（Spotter阶段一般为0）

        Returns:
            总损失（PDE损失 + 边界损失）
        """
        resample = kwargs.get('resample', False)
        return self.physics_loss_steady_state(data_GPU, resample=resample)

    def physics_loss_steady_state(self, data_GPU: Dict, resample: bool = False) -> torch.Tensor:
        """
        计算稳态PDE物理损失（Spotter专用）

        与GeneralPDENet.physics_loss的区别：
        - 支持每个epoch重新随机采样PDE内部点
        - 使用自适应学习率训练
        - Spotter阶段: 直接求解 R(U) = 0（不含伪时间项）

        Args:
            data_GPU: GPU数据字典
            resample: 是否重新采样PDE内部点

        Returns:
            总损失（PDE损失 + 边界损失）
        """
        # 如果需要重采样，则生成新的PDE内部点
        if resample and hasattr(self, '_n_pde_points'):
            x_train = self._resample_pde_points(self._n_pde_points, data_GPU['x_train'].device)
        else:
            x_train = data_GPU["x_train"]

        x_bd = data_GPU["x_bd"]
        u_bd = data_GPU["u_bd"]
        param = data_GPU["param"]

        # 获取模型预测（BaseNet 返回 (h, U)）
        _, U = self(x_train)

        # 获取物理参数
        Re = param[0].get("Re", 1.0)
        nu = param[0].get("nu", 1.0)

# auto code begin
# Config signature: {"F": ["0"], "L1": ["diff(u,t)-0.0001*diff(u,x,2)"], "L2": ["0"], "N": ["-5*u+5*u**3"]}
        # 提取物理量
        u = U[..., 0]

        # 计算一阶导数 (对所有空间维度)
        # x_train shape: [N, 2] where dim 0=x, dim 1=t
        du_x = self.gradients(u, x_train)[0][..., 0]  # ∂u/∂x
        du_t = self.gradients(u, x_train)[0][..., 1]  # ∂u/∂t (t是空间维度)

        # 计算二阶导数
        du_xx = self.gradients(du_x, x_train)[0][..., 0]  # ∂²u/∂x²

        # L1 算子: diff(u,t) - 0.0001*diff(u,x,2)
        L1 = [du_t - 0.0001*du_xx]

        # L2 算子
        L2 = [0]

        # F 算子
        F = [0]

        # N 算子
        N = [-5*u+5*u**3]

        # S 算子（源项）
        S = [0]

# auto code end

        # 稳态PDE残差: R(U) = L1 + L2*F + N - S = 0
        pde_loss = 0.0

        for i in range(self.config.n_eqs):
            # 计算PDE残差
            l1_term = L1[i]
            l2f_term = L2[i] * F[i]
            n_term = N[i]
            s_term = S[i]

            # PDE残差: L1 + L2*F + N - S = 0
            residual_i = l1_term + l2f_term + n_term - s_term
            pde_loss += torch.mean(residual_i**2)

        # 边界条件损失
        boundary_loss = self._compute_boundary_loss(data_GPU)
        boundary_loss_weight = 10.0

        # 总损失
        total_loss = pde_loss + boundary_loss_weight * boundary_loss
        return total_loss

    def _resample_pde_points(self, n_points: int, device) -> torch.Tensor:
        """
        重新随机采样PDE内部点（每个epoch）

        Args:
            n_points: 采样点数
            device: 设备

        Returns:
            新采样的坐标点 [n_points, n_dim]
        """
        # 从 config 读取 domain 信息
        x_domain = self.config.x_domain
        n_dim = self.config.n_dim

        # 为每个维度生成随机采样点
        coords = []
        for dim_idx in range(n_dim):
            x_min, x_max = x_domain[dim_idx]
            coord = torch.rand(n_points, 1, dtype=torch.float64, device=device) * (x_max - x_min) + x_min
            coords.append(coord)

        x_sampled = torch.cat(coords, dim=1)
        x_sampled.requires_grad_(True)
        return x_sampled

    def prepare_gpu_data(self, data: Dict, n_pde_points: Optional[int] = None) -> Dict:
        """
        为Spotter阶段准备GPU数据

        Args:
            data: 输入数据字典
            n_pde_points: PDE内部点数量（用于随机采样）

        Returns:
            gpu_data: GPU数据字典
        """
        gpu_data = {}

        # 如果指定了PDE点数，则随机采样；否则使用全部数据点
        if n_pde_points is not None:
            self._n_pde_points = n_pde_points
            x_train = self._resample_pde_points(n_pde_points, self.config.device)
        else:
            x_train = torch.tensor(
                data["x"], dtype=torch.float64, device=self.config.device, requires_grad=True
            )

        gpu_data["x_train"] = x_train

        # 边界条件数据
        boundary_gpu_data = self.prepare_boundary_gpu_data(data)
        gpu_data.update(boundary_gpu_data)

        # 提取边界点和值
        if "global_boundary_dict" in data and data["global_boundary_dict"]:
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
                gpu_data["x_bd"] = torch.zeros((0, self.config.n_dim), dtype=torch.float64, device=self.config.device)
                gpu_data["u_bd"] = torch.zeros((0, self.config.n_eqs), dtype=torch.float64, device=self.config.device)
        else:
            gpu_data["x_bd"] = torch.zeros((0, self.config.n_dim), dtype=torch.float64, device=self.config.device)
            gpu_data["u_bd"] = torch.zeros((0, self.config.n_eqs), dtype=torch.float64, device=self.config.device)

        # 物理参数
        gpu_data["param"] = [{"Re": 1.0, "nu": 1.0}]

        return gpu_data

    def train_spotter(self, data: Dict, n_epochs: int = 10000, lr_init: float = 1e-2,
                     n_pde_points: int = 10000, print_every: int = 1000):
        """
        Spotter阶段训练（带自适应学习率和随机采样）

        Args:
            data: 训练数据
            n_epochs: 训练轮数
            lr_init: 初始学习率
            n_pde_points: PDE内部点采样数
            print_every: 打印频率

        Returns:
            loss_history: 损失历史
        """
        print("=" * 80)
        print("Spotter阶段训练 (固定采样 + 自适应学习率 + float64)")
        print("=" * 80)
        print(f"训练轮数: {n_epochs}")
        print(f"PDE残差点数: {n_pde_points} (固定采样点)")
        print(f"初始学习率: {lr_init}")
        # Build network structure from config
        n_params = sum(p.numel() for p in self.parameters())
        in_dim = self.config.n_dim
        out_dim = self.config.n_eqs
        hidden_dims = self.config.hidden_dims
        net_structure = [in_dim] + list(hidden_dims) + [out_dim]
        print(f"网络结构: {net_structure} (参数数量: {n_params})")
        print(f"配置的 hidden_dims: {hidden_dims}")
        print("=" * 80)

        # 优化器
        optimizer = torch.optim.Adam(self.parameters(), lr=lr_init, betas=(0.999, 0.999))

        # 自适应学习率配置
        lr_thresholds = [1.0, 0.5, 1e-2, 1e-3, 1e-4]
        lr_values = [lr_init, 0.005, 0.003, 0.001, 0.0005, 0.0001]

        def adjust_learning_rate(current_loss):
            """根据loss自适应调整学习率（只降不升）"""
            current_lr = optimizer.param_groups[0]['lr']
            target_lr = lr_values[0]
            for i, threshold in enumerate(lr_thresholds):
                if current_loss < threshold:
                    target_lr = lr_values[i + 1]
            # 只允许学习率下降
            target_lr = min(target_lr, current_lr)
            if abs(current_lr - target_lr) > 1e-8:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = target_lr
                return True, target_lr
            return False, current_lr

        # 准备初始GPU数据（带边界条件）
        gpu_data_static = self.prepare_gpu_data(data, n_pde_points=None)

        # 保存PDE点数用于重采样
        self._n_pde_points = n_pde_points

        # 生成固定的PDE训练点（只生成一次）
        x_train_fixed = self._resample_pde_points(n_pde_points, self.config.device)
        gpu_data_static["x_train"] = x_train_fixed

        loss_history = []
        best_loss = float('inf')

        import time
        start_time = time.time()

        for epoch in range(n_epochs):
            self.train()
            optimizer.zero_grad()

            # ===== 注释掉重采样功能，改用固定采样点 =====
            # 每个epoch重新采样PDE内部点（已禁用）
            # x_train_new = self._resample_pde_points(n_pde_points, self.config.device)
            # gpu_data_static["x_train"] = x_train_new
            # ===== 现在使用固定的 x_train_fixed =====

            # 计算物理损失
            loss = self.physics_loss_steady_state(gpu_data_static, resample=False)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 记录损失
            current_loss = loss.item()
            loss_history.append(current_loss)

            # 自适应调整学习率
            lr_changed, current_lr = adjust_learning_rate(current_loss)

            # 保存最佳模型
            if current_loss < best_loss:
                best_loss = current_loss

            # 打印进度
            if (epoch + 1) % print_every == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:6d}/{n_epochs} | "
                      f"Loss: {current_loss:.6e} | "
                      f"LR: {current_lr:.6e} | "
                      f"Time: {elapsed:.1f}s", flush=True)

            # 学习率变化提示
            if lr_changed:
                print(f"   → 学习率调整至 {current_lr:.6e} (loss={current_loss:.6e})", flush=True)

        print("=" * 80)
        print(f"✓ Spotter训练完成! 总用时: {time.time() - start_time:.2f}s")
        print(f"  最佳Loss: {best_loss:.6e}")
        print("=" * 80)

        return loss_history

    @staticmethod
    def model_init(config, use_periodic: bool = False):
        """
        初始化Spotter模型（复用 BaseNet 结构）

        Args:
            config: 配置对象
            use_periodic: 是否使用周期性特征

        Returns:
            SpotterNet: 初始化的模型
        """
        model = SpotterNet(config, use_periodic=use_periodic).to(config.device)
        return model
