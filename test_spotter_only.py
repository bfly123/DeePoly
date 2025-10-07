#!/usr/bin/env python
"""测试 SpotterNet 训练结果"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.problem_solvers.general_pde_solver.core.spotter_init import SpotterNet
from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig
from src.problem_solvers.general_pde_solver.utils.data import GeneralPDEDataGenerator

print("\n" + "="*80)
print("测试 SpotterNet 训练（对比 standalone_pinn_ac_improved.py）")
print("="*80)

# 加载配置
case_dir = "cases/general_pde_cases/AC_equation"
config = GeneralPDEConfig(case_dir=case_dir)

# 生成数据
datagen = GeneralPDEDataGenerator(config)
data_train = datagen.generate_data("train")

# 创建 SpotterNet
spotter = SpotterNet(config, use_periodic=False).to(config.device)

print(f"\n网络结构:")
print(f"  Input dim: {spotter.in_dim}")
print(f"  Output dim: {spotter.out_dim}")
print(f"  Hidden dims: {config.hidden_dims}")
total_params = sum(p.numel() for p in spotter.parameters())
print(f"  Total parameters: {total_params}")

# 训练
print(f"\n开始训练...")
loss_history = spotter.train_spotter(
    data=data_train,
    n_epochs=1000,
    lr_init=0.01,
    n_pde_points=10000,
    print_every=200
)

# 评估训练结果
print(f"\n" + "="*80)
print("训练结果评估:")
print("="*80)

# 1. 检查最终loss
final_loss = loss_history[-1]
best_loss = min(loss_history)
print(f"Final loss: {final_loss:.6e}")
print(f"Best loss: {best_loss:.6e}")

# 2. 检查预测解的范围
with torch.no_grad():
    x_test = torch.tensor(data_train['x'], dtype=torch.float64, device=config.device)
    _, U_pred = spotter(x_test)
    U_np = U_pred.cpu().numpy()
    
    u_norm = np.linalg.norm(U_np)
    u_max = np.max(np.abs(U_np))
    u_mean = np.mean(U_np)
    u_std = np.std(U_np)
    
    print(f"\n预测解统计:")
    print(f"  ||U||: {u_norm:.6e}")
    print(f"  max|U|: {u_max:.6e}")
    print(f"  mean(U): {u_mean:.6e}")
    print(f"  std(U): {u_std:.6e}")

# 3. 对比期望值（参考 standalone 版本）
print(f"\n" + "="*80)
print("与 standalone_pinn_ac_improved.py 对比:")
print("="*80)
print(f"期望:")
print(f"  - max|U| 应该在 O(1) 量级 (例如 0.5 ~ 2.0)")
print(f"  - Best loss 应该达到 1e-5 ~ 1e-6 量级")

if u_max < 0.01:
    print(f"\n✗ 问题: max|U| = {u_max:.6e} 太小!")
    print(f"  可能原因:")
    print(f"  1. 边界条件设置不正确")
    print(f"  2. 初始条件缺失")
    print(f"  3. PDE方程实现有误")
elif u_max > 0.1:
    print(f"\n✓ max|U| = {u_max:.6e} 在合理范围内")
else:
    print(f"\n? max|U| = {u_max:.6e} 偏小但可接受")

if best_loss < 1e-5:
    print(f"✓ Best loss = {best_loss:.6e} 达到预期")
else:
    print(f"? Best loss = {best_loss:.6e} 可以继续训练降低")

print("="*80 + "\n")
