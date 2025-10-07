#!/usr/bin/env python
"""测试 Fitter 阶段边界条件的施加"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig
from src.problem_solvers.general_pde_solver.utils.data import GeneralPDEDataGenerator
from src.problem_solvers.general_pde_solver.core.net import GeneralPDENet
from src.problem_solvers.general_pde_solver.core.fitter import GeneralPDEFitter

print("\n" + "="*80)
print("测试 Fitter 阶段边界条件施加")
print("="*80)

# 1. 加载配置和数据
case_dir = "cases/general_pde_cases/AC_equation"
config = GeneralPDEConfig(case_dir=case_dir)
datagen = GeneralPDEDataGenerator(config)
data_train = datagen.generate_data("train")

print(f"\n[1] 数据生成:")
print(f"  内部点数: {data_train['x'].shape[0]}")

# 检查 boundary_segments_dict
if 'boundary_segments_dict' in data_train:
    print(f"\n  boundary_segments_dict 结构:")
    n_segments = len(data_train['boundary_segments_dict'])
    print(f"    总segment数: {n_segments}")

    total_dirichlet_points = 0
    for seg_idx, seg_bd in enumerate(data_train['boundary_segments_dict']):
        if seg_bd and isinstance(seg_bd, dict):  # 非空且是字典
            for var_idx, var_bd in seg_bd.items():
                if isinstance(var_bd, dict) and 'dirichlet' in var_bd:
                    if isinstance(var_bd['dirichlet']['x'], np.ndarray) and var_bd['dirichlet']['x'].shape[0] > 0:
                        n_pts = var_bd['dirichlet']['x'].shape[0]
                        total_dirichlet_points += n_pts
                        print(f"    Segment {seg_idx}, Var {var_idx}: {n_pts} Dirichlet 点")

    print(f"\n  ✓ 总 Dirichlet 边界点数: {total_dirichlet_points}")
else:
    print(f"  ✗ 警告: boundary_segments_dict 不存在")

# 2. 创建网络和 Fitter
print(f"\n[2] 创建 Fitter:")
model = GeneralPDENet(config, use_periodic=False).to(config.device)
fitter = GeneralPDEFitter(
    config,
    data=data_train,
    time_scheme="implicit_euler",
    visualizer=None
)

# 3. 初始化 Fitter（会调用 _add_boundary_conditions）
print(f"\n[3] Fitter 初始化 (会施加边界条件):")
fitter.fitter_init(model=model)
print(f"  ✓ Fitter 预编译完成")

# 4. 检查约束矩阵中的边界条件
print(f"\n[4] 检查约束矩阵:")
if hasattr(fitter, 'A') and hasattr(fitter, 'b'):
    print(f"  约束矩阵 A 包含 {len(fitter.A)} 个块")
    print(f"  右端项 b 包含 {len(fitter.b)} 个块")

    # 统计边界条件约束的行数
    boundary_rows = 0
    for i, (A_block, b_block) in enumerate(zip(fitter.A, fitter.b)):
        if A_block.shape[0] > 0:
            # 检查是否是边界条件约束（通常有权重10）
            # 边界条件约束的特征是：b值的量级与边界值一致
            if np.max(np.abs(b_block)) > 0.01:  # 边界值量级
                boundary_rows += A_block.shape[0]
                print(f"    Block {i}: {A_block.shape[0]} rows, b范围=[{b_block.min():.3f}, {b_block.max():.3f}]")

    print(f"\n  估计边界条件约束行数: {boundary_rows}")

    if boundary_rows > 0:
        print(f"  ✓ 边界条件已添加到约束矩阵")
    else:
        print(f"  ✗ 警告: 未检测到边界条件约束")
else:
    print(f"  ✗ Fitter 没有 A 和 b 属性")

# 5. 测试一步伪时间迭代，看边界条件是否保持
print(f"\n[5] 测试一步伪时间迭代:")

# 创建初始解（随机）
U_init = np.random.randn(data_train['x'].shape[0], config.n_eqs) * 0.1

print(f"  初始解统计:")
print(f"    ||U_init||: {np.linalg.norm(U_init):.6e}")
print(f"    range: [{U_init.min():.3f}, {U_init.max():.3f}]")

# 执行一步迭代
try:
    U_new, U_seg_new, coeffs = fitter.solve_steady_state(U_init=U_init, coeffs_init=None)

    print(f"\n  迭代后解统计:")
    print(f"    ||U_new||: {np.linalg.norm(U_new):.6e}")
    print(f"    range: [{U_new.min():.3f}, {U_new.max():.3f}]")

    # 检查边界点上的值
    # 需要找到对应边界条件的点
    if 'global_boundary_dict' in data_train:
        for var_idx, var_dict in data_train['global_boundary_dict'].items():
            if 'dirichlet' in var_dict:
                x_bd = var_dict['dirichlet']['x']
                u_bd_target = var_dict['dirichlet']['values']

                if x_bd.shape[0] > 0:
                    # 找到内部点中与边界点最接近的点
                    # 这只是近似检查
                    print(f"\n  边界值检查 (变量 {var_idx}):")
                    print(f"    目标边界值范围: [{u_bd_target.min():.3f}, {u_bd_target.max():.3f}]")

                    # 随机抽样几个边界点检查
                    sample_indices = np.random.choice(x_bd.shape[0], min(5, x_bd.shape[0]), replace=False)
                    for idx in sample_indices:
                        x_pt = x_bd[idx]
                        u_target = u_bd_target[idx, 0]
                        # 找到最近的内部点
                        dist = np.linalg.norm(data_train['x'] - x_pt, axis=1)
                        closest_idx = np.argmin(dist)
                        u_actual = U_new[closest_idx, var_idx]
                        print(f"    点 {idx}: target={u_target:.3f}, actual={u_actual:.3f}, dist={dist[closest_idx]:.4f}")

    print(f"\n  ✓ 伪时间步迭代成功完成")

except Exception as e:
    print(f"  ✗ 迭代失败: {e}")
    import traceback
    traceback.print_exc()

print("="*80 + "\n")
