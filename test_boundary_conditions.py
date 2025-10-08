"""
检查边界条件设置是否正确生效
特别关注下边界的Dirichlet条件
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig
from src.problem_solvers.time_pde_solver.utils.data import TimePDEDataGenerator

# 加载配置
case_dir = "cases/Time_pde_cases/New_test_2D"
config = TimePDEConfig(case_dir=case_dir)

print("=" * 80)
print("边界条件配置检查")
print("=" * 80)

# 显示配置信息
print("\n【配置信息】")
print(f"  问题类型: {config.problem_type}")
print(f"  空间维度: {config.n_dim}D")
print(f"  空间变量: {config.spatial_vars}")
print(f"  x域: {config.x_domain[0]}")
print(f"  y域: {config.x_domain[1]}")
print(f"  方程数: {config.n_eqs}")

# 显示边界条件配置
print("\n【边界条件配置】")
if hasattr(config, 'boundary_conditions'):
    for i, bc in enumerate(config.boundary_conditions):
        print(f"\n  边界条件 {i+1}:")
        print(f"    类型: {bc.get('type', 'N/A')}")
        if bc['type'] == 'periodic':
            print(f"    周期对: {bc.get('pairs', 'N/A')}")
        elif bc['type'] == 'dirichlet':
            print(f"    区域: {bc.get('region', 'N/A')}")
            print(f"    变量: {bc.get('var', 'N/A')}")
            print(f"    公式: {bc.get('formula', 'N/A')}")
        print(f"    点数: {bc.get('points', 'N/A')}")

# 生成训练数据
print("\n" + "=" * 80)
print("生成训练数据并检查边界条件")
print("=" * 80)

datagen = TimePDEDataGenerator(config)
data_train = datagen.generate_data("train")

print("\n【数据生成完成】")
print(f"  总训练点数: {len(data_train['x'])}")
print(f"  Segments: {len(data_train['x_segments'])}")

# 检查global_boundary_dict
print("\n【全局边界条件数据】")
if "global_boundary_dict" in data_train and data_train["global_boundary_dict"]:
    for var_idx in data_train["global_boundary_dict"]:
        print(f"\n  变量 {var_idx} ('{config.vars_list[var_idx]}'):")
        for bc_type in data_train["global_boundary_dict"][var_idx]:
            bc_data = data_train["global_boundary_dict"][var_idx][bc_type]
            if "x" in bc_data and bc_data["x"] is not None:
                x_bd = bc_data["x"]
                vals = bc_data["values"]
                print(f"    {bc_type}:")
                print(f"      点数: {len(x_bd)}")
                if len(x_bd) > 0:
                    # 确保是2D数组
                    if x_bd.ndim == 1:
                        print(f"      ⚠️  x_bd是1维数组，可能是空数组")
                    else:
                        print(f"      x范围: [{x_bd[:, 0].min():.3f}, {x_bd[:, 0].max():.3f}]")
                        print(f"      y范围: [{x_bd[:, 1].min():.3f}, {x_bd[:, 1].max():.3f}]")
                        print(f"      值范围: [{vals.min():.6f}, {vals.max():.6f}]")

# 检查segment级别的边界条件
print("\n【Segment级别的边界条件】")
if "boundary_segments_dict" in data_train:
    n_segments_with_bc = sum(1 for seg_bc in data_train["boundary_segments_dict"]
                              if seg_bc and any(seg_bc.values()))
    print(f"  有边界条件的segments数: {n_segments_with_bc} / {len(data_train['boundary_segments_dict'])}")

    # 详细查看每个segment
    for seg_idx, seg_bc in enumerate(data_train["boundary_segments_dict"]):
        if seg_bc and any(seg_bc.values()):
            print(f"\n  Segment {seg_idx}:")
            for var_idx in seg_bc:
                if seg_bc[var_idx]:
                    print(f"    变量 {var_idx}:")
                    for bc_type in seg_bc[var_idx]:
                        bc_info = seg_bc[var_idx][bc_type]
                        if "x" in bc_info and bc_info["x"] is not None:
                            print(f"      {bc_type}: {len(bc_info['x'])} 点")

# 可视化边界条件
print("\n" + "=" * 80)
print("可视化边界条件")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 所有训练点分布
ax1 = axes[0, 0]
x_all = data_train['x']
ax1.scatter(x_all[:, 0], x_all[:, 1], s=1, alpha=0.3, c='gray', label='训练点')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title(f'所有训练点分布 (总数: {len(x_all)})')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. 边界点分布
ax2 = axes[0, 1]
ax2.scatter(x_all[:, 0], x_all[:, 1], s=1, alpha=0.1, c='lightgray', label='训练点')

colors = ['red', 'blue', 'green', 'orange', 'purple']
if "global_boundary_dict" in data_train and data_train["global_boundary_dict"]:
    color_idx = 0
    for var_idx in data_train["global_boundary_dict"]:
        for bc_type in data_train["global_boundary_dict"][var_idx]:
            bc_data = data_train["global_boundary_dict"][var_idx][bc_type]
            if "x" in bc_data and bc_data["x"] is not None:
                x_bd = bc_data["x"]
                # 跳过空数组或1维数组
                if len(x_bd) > 0 and x_bd.ndim == 2:
                    color = colors[color_idx % len(colors)]
                    ax2.scatter(x_bd[:, 0], x_bd[:, 1], s=20, c=color,
                               label=f'{bc_type} ({len(x_bd)}点)', alpha=0.8, edgecolors='black', linewidth=0.5)
                    color_idx += 1

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('边界点分布')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. 下边界 (y=0) 的详细检查
ax3 = axes[1, 0]

# 提取下边界点 (y=0附近)
tolerance = 0.01
bottom_mask = np.abs(x_all[:, 1]) < tolerance
x_bottom = x_all[bottom_mask]

if len(x_bottom) > 0:
    ax3.scatter(x_bottom[:, 0], x_bottom[:, 1], s=10, c='lightblue', label='y≈0 的训练点', alpha=0.5)

# Dirichlet边界条件点
if "global_boundary_dict" in data_train and data_train["global_boundary_dict"]:
    for var_idx in data_train["global_boundary_dict"]:
        if "dirichlet" in data_train["global_boundary_dict"][var_idx]:
            bc_data = data_train["global_boundary_dict"][var_idx]["dirichlet"]
            if "x" in bc_data and bc_data["x"] is not None:
                x_bd = bc_data["x"]
                vals = bc_data["values"]
                # 跳过空数组或1维数组
                if len(x_bd) > 0 and x_bd.ndim == 2:
                    # 只显示y=0的点
                    bottom_bd_mask = np.abs(x_bd[:, 1]) < tolerance
                    x_bd_bottom = x_bd[bottom_bd_mask]
                    vals_bottom = vals[bottom_bd_mask]

                    if len(x_bd_bottom) > 0:
                        ax3.scatter(x_bd_bottom[:, 0], x_bd_bottom[:, 1], s=50, c='red',
                                   marker='x', label=f'Dirichlet边界点 ({len(x_bd_bottom)})',
                                   linewidths=2, zorder=5)
                        print(f"\n  下边界Dirichlet点数: {len(x_bd_bottom)}")
                        print(f"  x坐标范围: [{x_bd_bottom[:, 0].min():.3f}, {x_bd_bottom[:, 0].max():.3f}]")

ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('下边界 (y≈0) 详细视图')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_ylim(-0.05, 0.05)

# 4. Dirichlet边界值验证
ax4 = axes[1, 1]

# 理论值: u = x^2 * cos(pi * x)
x_theory = np.linspace(-1, 1, 200)
u_theory = x_theory**2 * np.cos(np.pi * x_theory)

ax4.plot(x_theory, u_theory, 'k-', linewidth=2, label='理论公式: x²cos(πx)', zorder=1)

# 实际边界值
if "global_boundary_dict" in data_train and data_train["global_boundary_dict"]:
    for var_idx in data_train["global_boundary_dict"]:
        if "dirichlet" in data_train["global_boundary_dict"][var_idx]:
            bc_data = data_train["global_boundary_dict"][var_idx]["dirichlet"]
            if "x" in bc_data and bc_data["x"] is not None:
                x_bd = bc_data["x"]
                vals = bc_data["values"]
                # 跳过空数组或1维数组
                if len(x_bd) > 0 and x_bd.ndim == 2:
                    # 只显示y=0的点
                    bottom_bd_mask = np.abs(x_bd[:, 1]) < tolerance
                    x_bd_bottom = x_bd[bottom_bd_mask]
                    vals_bottom = vals[bottom_bd_mask]

                    if len(x_bd_bottom) > 0:
                        # 按x坐标排序
                        sort_idx = np.argsort(x_bd_bottom[:, 0])
                        x_sorted = x_bd_bottom[sort_idx, 0]
                        vals_sorted = vals_bottom[sort_idx, 0]

                        ax4.scatter(x_sorted, vals_sorted, c='red', s=30,
                                  label=f'实际边界值 ({len(x_sorted)}点)', alpha=0.7, zorder=2)

                        # 计算误差
                        u_theory_at_bd = x_sorted**2 * np.cos(np.pi * x_sorted)
                        error = np.abs(vals_sorted - u_theory_at_bd)
                        max_error = np.max(error)
                        mean_error = np.mean(error)

                        print(f"\n【下边界Dirichlet条件验证】")
                        print(f"  理论公式: u = x² * cos(π*x)")
                        print(f"  最大误差: {max_error:.6e}")
                        print(f"  平均误差: {mean_error:.6e}")

                        if max_error > 1e-6:
                            print(f"  ⚠️  警告: 边界值与理论值存在较大偏差!")
                        else:
                            print(f"  ✓ 边界值与理论值一致")

ax4.set_xlabel('x')
ax4.set_ylabel('u')
ax4.set_title('下边界 Dirichlet 条件验证 (y=0)')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('boundary_conditions_check.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 可视化图保存至: boundary_conditions_check.png")
# plt.show()  # 注释掉，避免阻塞

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)
