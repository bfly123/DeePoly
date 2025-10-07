import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.problem_solvers.general_pde_solver.core.spotter_init import SpotterNet
from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig
from src.problem_solvers.general_pde_solver.utils.data import GeneralPDEDataGenerator
import torch
import numpy as np

case_dir = "cases/general_pde_cases/AC_equation"
config = GeneralPDEConfig(case_dir=case_dir)

datagen = GeneralPDEDataGenerator(config)
data_train = datagen.generate_data("train")

spotter = SpotterNet(config).to(config.device)

# 准备 GPU 数据
gpu_data = spotter.prepare_gpu_data(data_train, n_pde_points=100)

print("\n=== GPU数据检查 ===")
print(f"x_train shape: {gpu_data['x_train'].shape}")
print(f"x_bd shape: {gpu_data['x_bd'].shape}")
print(f"u_bd shape: {gpu_data['u_bd'].shape}")

if gpu_data['x_bd'].shape[0] > 0:
    print(f"\n边界条件点:")
    print(f"  x_bd[:5]: {gpu_data['x_bd'][:5]}")
    print(f"  u_bd[:5]: {gpu_data['u_bd'][:5]}")
else:
    print("\n✗ 边界条件数据为空！")

print(f"\n=== 检查 boundary loss 计算 ===")
with torch.no_grad():
    loss = spotter.physics_loss_steady_state(gpu_data, resample=False)
    print(f"Total loss: {loss.item():.6e}")

# 手动计算边界损失
if gpu_data['x_bd'].shape[0] > 0:
    with torch.no_grad():
        _, u_pred_bd = spotter(gpu_data['x_bd'])
        bc_loss = torch.mean((u_pred_bd - gpu_data['u_bd'])**2)
        print(f"Boundary loss (manual): {bc_loss.item():.6e}")
        print(f"u_pred_bd range: [{u_pred_bd.min():.3f}, {u_pred_bd.max():.3f}]")
        print(f"u_bd target: [{gpu_data['u_bd'].min():.3f}, {gpu_data['u_bd'].max():.3f}]")
else:
    print("无法计算边界损失 - 边界数据为空")
