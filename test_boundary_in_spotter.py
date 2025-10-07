#!/usr/bin/env python
"""测试 SpotterNet 中边界条件数据的使用"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.problem_solvers.general_pde_solver.core.spotter_init import SpotterNet
from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig
from src.problem_solvers.general_pde_solver.utils.data import GeneralPDEDataGenerator

print("\n" + "="*80)
print("测试 SpotterNet 边界条件数据流")
print("="*80)

# 加载配置
case_dir = "cases/general_pde_cases/AC_equation"
config = GeneralPDEConfig(case_dir=case_dir)

# 生成数据
print("\n[1] 生成数据...")
datagen = GeneralPDEDataGenerator(config)
data_train = datagen.generate_data("train")

print(f"  内部点: {data_train['x'].shape}")

# 检查 global_boundary_dict
if 'global_boundary_dict' in data_train:
    print(f"  global_boundary_dict keys: {list(data_train['global_boundary_dict'].keys())}")
    for var_idx, var_dict in data_train['global_boundary_dict'].items():
        print(f"    变量 {var_idx}:")
        for bc_type, bc_data in var_dict.items():
            if 'x' in bc_data and 'values' in bc_data:
                if isinstance(bc_data['x'], np.ndarray) and bc_data['x'].size > 0:
                    print(f"      {bc_type}: x.shape={bc_data['x'].shape}, values.shape={bc_data['values'].shape}")

# 创建 SpotterNet 并准备 GPU 数据
print("\n[2] 准备 SpotterNet GPU 数据...")
spotter = SpotterNet(config, use_periodic=False).to(config.device)

gpu_data = spotter.prepare_gpu_data(data_train, n_pde_points=1000)

print(f"  GPU data keys: {list(gpu_data.keys())}")
if 'x_bd' in gpu_data:
    print(f"  x_bd shape: {gpu_data['x_bd'].shape}")
    print(f"  u_bd shape: {gpu_data['u_bd'].shape}")
    print(f"  ✓ 边界数据成功加载到 GPU")

    # 检查边界值的范围
    u_bd_vals = gpu_data['u_bd'].cpu().numpy()
    print(f"\n  边界值统计:")
    print(f"    min: {np.min(u_bd_vals):.4f}")
    print(f"    max: {np.max(u_bd_vals):.4f}")
    print(f"    mean: {np.mean(u_bd_vals):.4f}")
else:
    print(f"  ✗ 边界数据未加载")

# 计算边界损失
print("\n[3] 计算边界损失...")
boundary_loss = spotter._compute_boundary_loss(gpu_data)
print(f"  Boundary loss: {boundary_loss.item():.6e}")

if boundary_loss.item() > 0:
    print(f"  ✓ 边界损失非零，边界条件正在起作用")
else:
    print(f"  ? 边界损失为零")

print("="*80 + "\n")
