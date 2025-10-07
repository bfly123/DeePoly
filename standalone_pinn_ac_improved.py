"""
改进版PINN求解Allen-Cahn方程
基于kimy-de/pinns的关键改进：
1. 更大的网络容量 (4×128)
2. Weight Normalization
3. 固定网格采样
4. 强制边界值约束 u(±1,t)=-1
5. 更好的优化器参数
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time


# ======================= 改进的神经网络定义 =======================
class LinearBlock(nn.Module):
    """带Weight Normalization的线性块"""
    def __init__(self, in_nodes, out_nodes):
        super().__init__()
        self.layer = nn.utils.weight_norm(nn.Linear(in_nodes, out_nodes, dtype=torch.float64), dim=0)

    def forward(self, x):
        return torch.tanh(self.layer(x))


class PINN_AC_Improved(nn.Module):
    """改进的Physics-Informed Neural Network"""

    def __init__(self, layers=[2, 128, 128, 128, 128, 1]):
        super().__init__()

        # 输入层使用weight_norm (float64)
        self.input_layer = nn.utils.weight_norm(nn.Linear(layers[0], layers[1], dtype=torch.float64), dim=0)

        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(layers) - 2):
            self.hidden_layers.append(LinearBlock(layers[i], layers[i+1]))

        # 输出层不使用weight_norm (float64)
        self.output_layer = nn.Linear(layers[-2], layers[-1], dtype=torch.float64)

        # Xavier初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x, t):
        """
        Args:
            x: [N, 1]
            t: [N, 1]
        Returns:
            u: [N, 1]
        """
        inputs = torch.cat([x, t], dim=1)

        # 输入层
        h = torch.tanh(self.input_layer(inputs))

        # 隐藏层
        for layer in self.hidden_layers:
            h = layer(h)

        # 输出层
        u = self.output_layer(h)
        return u

    def compute_derivatives(self, x, t):
        """计算导数"""
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = self.forward(x, t)

        # 一阶导数
        du = torch.autograd.grad(
            u, [x, t],
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )
        du_dx, du_dt = du[0], du[1]

        # 二阶导数
        d2u_dx2 = torch.autograd.grad(
            du_dx, x,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True
        )[0]

        return u, du_dt, du_dx, d2u_dx2


# ======================= 改进的数据生成（随机采样版本）=======================
def generate_training_data(n_interior=10000, n_boundary=1000, device='cpu'):
    """
    生成随机采样的训练数据（恢复原始随机采样策略）

    Returns:
        x_bc, t_bc, u_bc: 边界和初始条件数据
        x_pde, t_pde: PDE残差计算点（随机采样）
    """
    # PDE内部点：随机采样
    x_pde = (torch.rand(n_interior, 1, dtype=torch.float64, device=device) * 2 - 1)  # [-1, 1]
    t_pde = torch.rand(n_interior, 1, dtype=torch.float64, device=device)            # [0, 1]

    # 初始条件点 (t = 0)
    x_init = torch.rand(n_boundary, 1, dtype=torch.float64, device=device) * 2 - 1
    t_init = torch.zeros(n_boundary, 1, dtype=torch.float64, device=device)
    u_init = (x_init**2) * torch.cos(np.pi * x_init)

    # 周期边界点 (x = ±1)
    t_boundary = torch.rand(n_boundary, 1, dtype=torch.float64, device=device)
    x_left = -torch.ones(n_boundary, 1, dtype=torch.float64, device=device)
    x_right = torch.ones(n_boundary, 1, dtype=torch.float64, device=device)
    u_boundary = -torch.ones((2 * n_boundary, 1), dtype=torch.float64, device=device)  # u(±1,t) = -1

    # 合并所有边界条件
    x_bc = torch.cat([x_init, x_left, x_right], dim=0)
    t_bc = torch.cat([t_init, t_boundary, t_boundary], dim=0)
    u_bc = torch.cat([u_init, u_boundary], dim=0)

    return x_bc, t_bc, u_bc, x_pde, t_pde


# ======================= 损失函数 =======================
def ac_equation(u, x, t, epsilon=0.0001):
    """Allen-Cahn方程残差"""
    x.requires_grad_(True)
    t.requires_grad_(True)

    # 计算导数
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # PDE: u_t - ε*u_xx - 5u + 5u³ = 0
    residual = u_t - epsilon * u_xx + 5*u**3 - 5*u
    return residual


# ======================= 训练函数 =======================
def train_pinn_improved(model, x_bc, t_bc, u_bc, x_pde, t_pde,
                        n_epochs=20000, lr=1e-2, device='cpu', print_every=1000):
    """
    改进的训练流程（带自适应学习率衰减）
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.999, 0.999))

    # 设置requires_grad (BC固定，PDE点每epoch重采样)
    x_bc = x_bc.clone().detach().requires_grad_(False)
    t_bc = t_bc.clone().detach().requires_grad_(False)
    u_bc = u_bc.clone().detach().requires_grad_(False)

    # 记录PDE点数量用于每epoch重采样
    n_pde_points = len(x_pde)

    loss_history = []
    pde_loss_history = []
    bc_loss_history = []

    # 自适应学习率配置（参考base_net.py）
    # 从1e-2启动，快速衰减到合适范围
    lr_thresholds = [1.0, 0.5, 1e-2, 1e-3, 1e-4]
    lr_values = [lr, 0.005, 0.003, 0.001, 0.0005, 0.0001]

    def adjust_learning_rate(optimizer, current_loss):
        """根据loss自适应调整学习率（只降不升，避免振荡）"""
        current_lr = optimizer.param_groups[0]['lr']

        # 从高到低检查阈值，选择最小的满足条件的学习率
        target_lr = lr_values[0]  # 默认最大学习率
        for i, threshold in enumerate(lr_thresholds):
            if current_loss < threshold:
                target_lr = lr_values[i + 1]

        # 只允许学习率下降，不允许上升（避免振荡）
        target_lr = min(target_lr, current_lr)

        # 如果学习率变化显著，则应用调整
        if abs(current_lr - target_lr) > 1e-8:
            for param_group in optimizer.param_groups:
                param_group['lr'] = target_lr
            return True, target_lr
        return False, current_lr

    print("=" * 80, flush=True)
    print("改进版PINN训练 (随机采样 + 自适应学习率 + float64)", flush=True)
    print("=" * 80, flush=True)
    print(f"训练轮数: {n_epochs}", flush=True)
    print(f"PDE残差点数: {len(x_pde)} (每epoch随机重采样)", flush=True)
    print(f"边界/初值点数: {len(x_bc)}", flush=True)
    print(f"初始学习率: {lr}", flush=True)
    print(f"学习率衰减阈值: {lr_thresholds}", flush=True)
    print(f"学习率取值: {lr_values}", flush=True)
    print(f"优化器: Adam(betas=(0.999, 0.999))", flush=True)
    print("=" * 80, flush=True)
    print("开始训练...", flush=True)

    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # 每个epoch重新采样PDE内部点
        x_pde = (torch.rand(n_pde_points, 1, dtype=torch.float64, device=device) * 2 - 1)
        t_pde = torch.rand(n_pde_points, 1, dtype=torch.float64, device=device)
        x_pde.requires_grad_(True)
        t_pde.requires_grad_(True)

        # 边界条件损失
        u_bc_pred = model(x_bc, t_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc)**2)

        # PDE残差损失
        u_pde = model(x_pde, t_pde)
        residual = ac_equation(u_pde, x_pde, t_pde)
        loss_pde = torch.mean(residual**2)

        # 总损失
        loss = loss_pde + 10*loss_bc

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录
        current_loss = loss.item()
        loss_history.append(current_loss)
        pde_loss_history.append(loss_pde.item())
        bc_loss_history.append(loss_bc.item())

        # 自适应调整学习率
        lr_changed, current_lr = adjust_learning_rate(optimizer, current_loss)

        # 保存最佳模型
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), 'pinn_ac_improved_best.pth')

        # 打印
        if (epoch + 1) % print_every == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:6d}/{n_epochs} | "
                  f"Loss: {current_loss:.6e} | "
                  f"PDE: {loss_pde.item():.6e} | "
                  f"BC: {loss_bc.item():.6e} | "
                  f"LR: {current_lr:.6e} | "
                  f"Time: {elapsed:.1f}s", flush=True)

        # 学习率变化时打印提示
        if lr_changed:
            print(f"   → 学习率调整至 {current_lr:.6e} (loss={current_loss:.6e})", flush=True)

    print("=" * 80)
    print(f"✓ 训练完成! 总用时: {time.time() - start_time:.2f}s")
    print(f"  最佳Loss: {best_loss:.6e}")
    print("=" * 80)

    return loss_history, pde_loss_history, bc_loss_history


# ======================= 可视化 =======================
def plot_results(model, device='cpu'):
    """生成结果可视化"""
    model.eval()

    # 生成测试网格 (使用float64匹配模型)
    x = torch.linspace(-1, 1, 256, dtype=torch.float64, device=device).reshape(-1, 1)
    t = torch.linspace(0, 1, 100, dtype=torch.float64, device=device).reshape(-1, 1)

    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    with torch.no_grad():
        u_pred = model(x_flat, t_flat)

    U = u_pred.reshape(256, 100).cpu().numpy()
    X_np = X.cpu().numpy()
    T_np = T.cpu().numpy()

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 时空图
    ax1 = axes[0]
    c1 = ax1.contourf(T_np.T, X_np.T, U.T, levels=50, cmap='RdYlBu_r')
    plt.colorbar(c1, ax=ax1)
    ax1.set_xlabel('t', fontsize=12)
    ax1.set_ylabel('x', fontsize=12)
    ax1.set_title('Improved PINN Solution u(x,t)', fontsize=14, fontweight='bold')

    # 不同时刻
    ax2 = axes[1]
    for t_idx in [0, 25, 50, 75, 99]:
        ax2.plot(x.cpu().numpy(), U[:, t_idx],
                label=f't={t_idx/99:.2f}', linewidth=2)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('u', fontsize=12)
    ax2.set_title('Solution at Different Times', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pinn_ac_improved_solution.png', dpi=150, bbox_inches='tight')
    print("✓ 解的可视化保存至: pinn_ac_improved_solution.png")
    plt.close()


def plot_loss(loss_history, pde_loss_history, bc_loss_history):
    """绘制损失曲线"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.semilogy(loss_history, 'b-', linewidth=1.5, label='Total Loss', alpha=0.8)
    ax.semilogy(pde_loss_history, 'r-', linewidth=1.5, label='PDE Loss', alpha=0.6)
    ax.semilogy(bc_loss_history, 'g-', linewidth=1.5, label='BC Loss', alpha=0.6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Training Loss History (Improved)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pinn_ac_improved_loss.png', dpi=150, bbox_inches='tight')
    print("✓ 损失曲线保存至: pinn_ac_improved_loss.png")
    plt.close()


# ======================= 主程序 =======================
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}\n", flush=True)

    # 创建改进的模型
    print("创建模型...", flush=True)
    model = PINN_AC_Improved(layers=[2, 128, 128, 128, 128, 1]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params}\n", flush=True)

    # 生成训练数据（随机采样）
    print("生成随机采样训练数据...", flush=True)
    x_bc, t_bc, u_bc, x_pde, t_pde = generate_training_data(
        n_interior=10000, n_boundary=1000, device=device
    )
    print(f"✓ 数据生成完成", flush=True)
    print(f"  边界/初值点: {len(x_bc)}", flush=True)
    print(f"  PDE残差点: {len(x_pde)}\n", flush=True)

    # 训练
    loss_history, pde_loss_history, bc_loss_history = train_pinn_improved(
        model, x_bc, t_bc, u_bc, x_pde, t_pde,
        n_epochs=20000,  # 可以增加到200000
        lr=1e-2,  # 从1e-2开始，自适应衰减
        device=device,
        print_every=1000
    )

    # 加载最佳模型
    model.load_state_dict(torch.load('pinn_ac_improved_best.pth'))

    # 可视化
    print("\n生成可视化...")
    plot_loss(loss_history, pde_loss_history, bc_loss_history)
    plot_results(model, device=device)

    print("\n" + "=" * 80)
    print("✓ 所有任务完成!")
    print("=" * 80)
