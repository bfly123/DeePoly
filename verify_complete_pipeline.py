#!/usr/bin/env python
"""验证完整的 General PDE Solver 流程"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.problem_solvers.general_pde_solver.core.spotter_init import SpotterNet
from src.problem_solvers.general_pde_solver.core.net import GeneralPDENet
from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig
from src.problem_solvers.general_pde_solver.utils.data import GeneralPDEDataGenerator

print("\n" + "="*80)
print("验证完整的 General PDE Solver Pipeline")
print("="*80)

# 1. 加载配置
print("\n[1/5] 加载配置...")
case_dir = "cases/general_pde_cases/AC_equation"
config = GeneralPDEConfig(case_dir=case_dir)

print(f"  问题类型: {config.problem_name}")
print(f"  空间维度: {config.spatial_vars} (t作为空间维度)")
print(f"  空间域: {config.x_domain}")
print(f"  方程: L1 = {config.eq['L1']}, N = {config.eq['N']}")
print("  ✓ 配置加载成功")

# 2. 生成数据（包括边界条件）
print("\n[2/5] 生成训练数据（包括边界/初始条件）...")
datagen = GeneralPDEDataGenerator(config)
data_train = datagen.generate_data("train")

print(f"  内部点: {data_train['x'].shape[0]} points in {data_train['x'].shape[1]}D")
if 'x_bd' in data_train and data_train['x_bd'].shape[0] > 0:
    print(f"  边界点: {data_train['x_bd'].shape[0]} points")
    print(f"  边界值: {data_train['u_bd'].shape}")
    print("  ✓ 边界/初始条件数据生成成功")
else:
    print("  ✗ 警告: 边界条件数据为空")

# 3. 创建并训练 SpotterNet
print("\n[3/5] 创建并训练 SpotterNet（初始化阶段）...")
spotter = SpotterNet(config, use_periodic=False).to(config.device)

total_params = sum(p.numel() for p in spotter.parameters())
print(f"  网络参数: {total_params}")
print(f"  架构: Input({spotter.in_dim}) → Hidden{config.hidden_dims} → Output({spotter.out_dim})")

print("\n  训练 SpotterNet (500 epochs)...")
loss_history = spotter.train_spotter(
    data=data_train,
    n_epochs=500,
    lr_init=0.01,
    n_pde_points=10000,
    print_every=100
)

best_loss = min(loss_history)
final_loss = loss_history[-1]
print(f"\n  训练完成:")
print(f"    Best loss: {best_loss:.6e}")
print(f"    Final loss: {final_loss:.6e}")

# 4. 检查 SpotterNet 预测解
print("\n[4/5] 评估 SpotterNet 初始解质量...")
with torch.no_grad():
    x_test = torch.tensor(data_train['x'], dtype=torch.float64, device=config.device)
    _, U_spotter = spotter(x_test)
    U_np = U_spotter.cpu().numpy()

    u_norm = np.linalg.norm(U_np)
    u_max = np.max(np.abs(U_np))
    u_mean = np.mean(U_np)
    u_std = np.std(U_np)

    print(f"  ||U||:    {u_norm:.6e}")
    print(f"  max|U|:  {u_max:.6e}")
    print(f"  mean(U): {u_mean:.6e}")
    print(f"  std(U):  {u_std:.6e}")

    if u_max > 0.1:
        print("  ✓ 解的幅度合理 (O(1) 量级)")
    else:
        print(f"  ✗ 警告: 解的幅度偏小")

# 5. 创建 GeneralPDENet 并测试参数迁移
print("\n[5/5] 测试网络参数迁移 (SpotterNet → GeneralPDENet)...")
fitter = GeneralPDENet(config, use_periodic=False).to(config.device)

fitter_params = sum(p.numel() for p in fitter.parameters())
print(f"  GeneralPDENet 参数: {fitter_params}")

if total_params == fitter_params:
    print("  ✓ 参数数量一致")
else:
    print(f"  ✗ 参数数量不一致: {total_params} vs {fitter_params}")

# 迁移参数
print("\n  执行参数迁移...")
fitter.load_state_dict(spotter.state_dict())

# 验证迁移正确性
with torch.no_grad():
    _, U_fitter = fitter(x_test)
    U_fitter_np = U_fitter.cpu().numpy()

    diff = np.linalg.norm(U_fitter_np - U_np)
    rel_diff = diff / (np.linalg.norm(U_np) + 1e-12)

    print(f"\n  迁移后验证:")
    print(f"    ||U_fitter - U_spotter||: {diff:.6e}")
    print(f"    相对误差: {rel_diff:.6e}")

    if diff < 1e-10:
        print("  ✓ 参数迁移完全正确")
    else:
        print(f"  ✗ 参数迁移有误差")

# 总结
print("\n" + "="*80)
print("Pipeline 验证总结")
print("="*80)

status = []
status.append(("配置加载", True))
status.append(("边界条件数据生成", 'x_bd' in data_train and data_train['x_bd'].shape[0] > 0))
status.append(("SpotterNet 训练", best_loss < 1e-4))
status.append(("解的幅度", u_max > 0.1))
status.append(("网络架构一致性", total_params == fitter_params))
status.append(("参数迁移正确性", diff < 1e-10))

all_passed = all(s[1] for s in status)

for name, passed in status:
    symbol = "✓" if passed else "✗"
    print(f"  {symbol} {name}")

if all_passed:
    print("\n  🎉 所有测试通过! Pipeline 工作正常.")
else:
    print("\n  ⚠️  部分测试未通过，请检查上述项目.")

print("="*80 + "\n")
