"""
测试独立的Spotter Net
"""
import torch
import numpy as np
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.problem_solvers.general_pde_solver.core.spotter_net import SpotterNet
from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig

if __name__ == "__main__":
    print("=" * 80)
    print("测试 SpotterNet - 独立启动步网络")
    print("=" * 80)

    # 加载配置
    case_path = "cases/Time_pde_cases/Allen_Cahn/AC_equation_100_0.01"
    print(f"\n案例: {case_path}")

    config = GeneralPDEConfig(case_dir=case_path)

    # 显示配置信息
    print(f"\n配置信息:")
    print(f"  空间维度: {config.n_dim}")
    print(f"  方程数: {config.n_eqs}")
    print(f"  设备: {config.device}")
    print(f"  空间变量: {config.spatial_vars}")

    # 生成训练数据 (SpotterNet需要[x, t]坐标，不只是空间坐标)
    print(f"\n生成训练数据...")

    # 为时间依赖PDE生成[x, t]训练数据
    n_spatial = 100
    n_time = 10

    # 从config获取空间和时间域
    x_min, x_max = config.x_domain[0][0], config.x_domain[0][1]
    t_min, t_max = 0.0, float(config.T)

    # 生成网格点
    x_spatial = np.linspace(x_min, x_max, n_spatial)
    t_temporal = np.linspace(t_min, t_max, n_time)

    # 创建[x, t]网格
    X, T = np.meshgrid(x_spatial, t_temporal)
    x_train = np.column_stack([X.ravel(), T.ravel()])

    # 初始条件 u(x, 0) = x^2 * cos(pi*x)
    u_initial = x_spatial**2 * np.cos(np.pi * x_spatial)

    # 简化测试：暂时不使用边界条件（SpotterNet会通过prepare_gpu_data处理）
    data_train = {
        'x': x_train,
        'u': np.zeros((len(x_train), 1)),  # 占位符
        'global_boundary_dict': {}  # 空字典，不强制边界条件
    }

    print(f"  训练点数: {len(data_train['x'])}")
    print(f"  坐标维度: {data_train['x'].shape}")

    # 创建SpotterNet模型
    print(f"\n创建SpotterNet模型...")
    spotter_model = SpotterNet.model_init(config, layers=[2, 128, 128, 128, 128, 1])
    total_params = sum(p.numel() for p in spotter_model.parameters())
    print(f"  模型参数量: {total_params}")

    # 检查数据类型
    print(f"\n检查模型数据类型:")
    for name, param in spotter_model.named_parameters():
        print(f"  {name}: {param.dtype}")
        break

    # 训练Spotter阶段
    print(f"\n开始Spotter训练...")
    loss_history = spotter_model.train_spotter(
        data=data_train,
        n_epochs=5000,
        lr_init=1e-2,
        n_pde_points=10000,
        print_every=1000
    )

    # 评估最终预测
    print(f"\n评估最终预测...")
    spotter_model.eval()
    with torch.no_grad():
        x_test = torch.tensor(
            data_train['x'], dtype=torch.float64, device=config.device
        )
        U_pred = spotter_model(x_test)
        u_norm = torch.norm(U_pred).item()
        u_max = torch.max(torch.abs(U_pred)).item()
        u_min = torch.min(U_pred).item()

        print(f"  解的范数: ||U|| = {u_norm:.6e}")
        print(f"  解的范围: [{u_min:.6e}, {u_max:.6e}]")

    print("\n" + "=" * 80)
    print("✓ SpotterNet测试完成!")
    print("=" * 80)

    # 绘制损失曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.semilogy(loss_history, 'b-', linewidth=1.5, alpha=0.8)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (log scale)', fontsize=12)
        plt.title('SpotterNet Training Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('spotter_net_loss.png', dpi=150, bbox_inches='tight')
        print("\n✓ 损失曲线已保存至: spotter_net_loss.png")
        plt.close()
    except Exception as e:
        print(f"\n⚠ 绘图失败: {e}")
