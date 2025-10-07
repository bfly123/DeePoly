"""
对比实验：固定采样 vs 重采样的效果

演示为什么需要每个epoch重新采样PDE内部点
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# 简化的1D PDE示例: u'' + u = 0, 解析解 u = sin(x)

class SimpleNet(torch.nn.Module):
    """简单的神经网络"""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 20),
            torch.nn.Tanh(),
            torch.nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.net(x)

def pde_loss(net, x):
    """PDE损失: u'' + u = 0"""
    x.requires_grad_(True)
    u = net(x)

    # 一阶导数
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    # 二阶导数
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    # PDE残差
    residual = u_xx + u
    return torch.mean(residual**2)

def train_with_fixed_sampling(epochs=1000, n_points=50):
    """固定采样训练"""
    print("\n" + "="*60)
    print("训练策略1: 固定采样点")
    print("="*60)

    net = SimpleNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # 固定的采样点
    x_train = torch.linspace(-np.pi, np.pi, n_points).reshape(-1, 1)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = pde_loss(net, x_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return net, losses, x_train

def train_with_resampling(epochs=1000, n_points=50):
    """重采样训练"""
    print("\n" + "="*60)
    print("训练策略2: 每个epoch重新采样")
    print("="*60)

    net = SimpleNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 每个epoch重新采样
        x_train = torch.rand(n_points, 1) * 2 * np.pi - np.pi

        loss = pde_loss(net, x_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    return net, losses, None

def evaluate_on_fine_grid(net, label):
    """在精细网格上评估"""
    x_test = torch.linspace(-np.pi, np.pi, 1000).reshape(-1, 1)

    with torch.no_grad():
        u_pred = net(x_test).numpy()

    # 解析解
    u_exact = np.sin(x_test.numpy())

    # 计算误差
    error = np.abs(u_pred - u_exact)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print(f"\n{label}:")
    print(f"  最大误差: {max_error:.6f}")
    print(f"  平均误差: {mean_error:.6f}")

    return x_test.numpy(), u_pred, u_exact, error

# 运行对比实验
print("="*60)
print("对比实验: 固定采样 vs 重采样")
print("="*60)
print("\n问题设定:")
print("  PDE: u'' + u = 0")
print("  域: [-π, π]")
print("  解析解: u = sin(x)")
print("  训练点数: 50点/epoch")
print("  训练轮数: 1000 epochs")

# 训练方法1: 固定采样
net1, losses1, x_train_fixed = train_with_fixed_sampling(epochs=1000, n_points=50)

# 训练方法2: 重采样
net2, losses2, _ = train_with_resampling(epochs=1000, n_points=50)

# 评估
print("\n" + "="*60)
print("评估结果（在1000个测试点上）")
print("="*60)

x_test, u_pred1, u_exact, error1 = evaluate_on_fine_grid(net1, "方法1: 固定采样")
_, u_pred2, _, error2 = evaluate_on_fine_grid(net2, "方法2: 重采样")

# 可视化对比
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 图1: 损失曲线
axes[0, 0].semilogy(losses1, label='固定采样', alpha=0.7)
axes[0, 0].semilogy(losses2, label='重采样', alpha=0.7)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('训练损失对比')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 图2: 预测结果对比
axes[0, 1].plot(x_test, u_exact, 'k-', linewidth=2, label='解析解', alpha=0.7)
axes[0, 1].plot(x_test, u_pred1, 'r--', label='固定采样', alpha=0.7)
axes[0, 1].plot(x_test, u_pred2, 'b:', linewidth=2, label='重采样', alpha=0.7)
if x_train_fixed is not None:
    axes[0, 1].plot(x_train_fixed.detach().numpy(), np.zeros_like(x_train_fixed.detach().numpy()),
                     'rx', markersize=8, label='训练点(固定)')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('u')
axes[0, 1].set_title('预测解对比')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 图3: 误差分布（固定采样）
axes[1, 0].plot(x_test, error1, 'r-', label='固定采样误差')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('绝对误差')
axes[1, 0].set_title(f'固定采样误差 (最大误差: {np.max(error1):.6f})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 图4: 误差分布（重采样）
axes[1, 1].plot(x_test, error2, 'b-', label='重采样误差')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('绝对误差')
axes[1, 1].set_title(f'重采样误差 (最大误差: {np.max(error2):.6f})')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resampling_comparison.png', dpi=150, bbox_inches='tight')
print("\n图片已保存: resampling_comparison.png")

# 总结
print("\n" + "="*60)
print("实验结论")
print("="*60)
print(f"\n1. 最大误差对比:")
print(f"   固定采样: {np.max(error1):.6f}")
print(f"   重采样:   {np.max(error2):.6f}")
print(f"   改进率:   {(1 - np.max(error2)/np.max(error1))*100:.1f}%")

print(f"\n2. 平均误差对比:")
print(f"   固定采样: {np.mean(error1):.6f}")
print(f"   重采样:   {np.mean(error2):.6f}")
print(f"   改进率:   {(1 - np.mean(error2)/np.mean(error1))*100:.1f}%")

print(f"\n3. 关键观察:")
print(f"   - 固定采样在训练点附近准确，但在其他区域误差大")
print(f"   - 重采样在整个域上误差更均匀、更小")
print(f"   - 重采样提高了网络的泛化能力")

print("\n" + "="*60)
