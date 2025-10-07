#!/usr/bin/env python
"""简单测试：验证网络参数迁移"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.problem_solvers.general_pde_solver.core.spotter_init import SpotterNet
from src.problem_solvers.general_pde_solver.core.net import GeneralPDENet
from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig

print("\n" + "="*80)
print("Simple Network Transfer Test")
print("="*80)

# 创建配置
case_dir = "cases/general_pde_cases/AC_equation"
config = GeneralPDEConfig(case_dir=case_dir)

print(f"\n1. Creating SpotterNet...")
spotter = SpotterNet(config, use_periodic=False).to(config.device)
print(f"   - Input dim: {spotter.in_dim}")
print(f"   - Output dim: {spotter.out_dim}")
print(f"   - Hidden dims: {config.hidden_dims}")

print(f"\n2. Creating GeneralPDENet...")
general_net = GeneralPDENet(config, use_periodic=False).to(config.device)
print(f"   - Input dim: {general_net.in_dim}")
print(f"   - Output dim: {general_net.out_dim}")
print(f"   - Hidden dims: {config.hidden_dims}")

print(f"\n3. Testing network architecture compatibility...")
spotter_params = sum(p.numel() for p in spotter.parameters())
general_params = sum(p.numel() for p in general_net.parameters())
print(f"   - SpotterNet parameters: {spotter_params}")
print(f"   - GeneralPDENet parameters: {general_params}")

if spotter_params == general_params:
    print("   ✓ Parameter counts match!")
else:
    print(f"   ✗ Parameter counts mismatch!")
    sys.exit(1)

print(f"\n4. Testing forward pass compatibility...")
x_test = torch.randn(10, config.n_dim, dtype=torch.float64, device=config.device, requires_grad=True)

with torch.no_grad():
    h_spotter, u_spotter = spotter(x_test)
    h_general, u_general = general_net(x_test)

print(f"   - SpotterNet output shape: {u_spotter.shape}")
print(f"   - GeneralPDENet output shape: {u_general.shape}")

if u_spotter.shape == u_general.shape:
    print("   ✓ Output shapes match!")
else:
    print(f"   ✗ Output shapes mismatch!")
    sys.exit(1)

print(f"\n5. Testing parameter transfer...")
# 保存 SpotterNet 参数
spotter_state = spotter.state_dict()

# 加载到 GeneralPDENet
general_net.load_state_dict(spotter_state)
print("   ✓ Parameters transferred successfully!")

# 验证迁移
with torch.no_grad():
    h_after, u_after = general_net(x_test)
    
# 比较输出（应该完全相同）
diff = torch.abs(u_spotter - u_after).max().item()
print(f"   - Max difference after transfer: {diff:.6e}")

if diff < 1e-12:
    print("   ✓ Outputs match perfectly after transfer!")
else:
    print(f"   ✗ Outputs differ after transfer!")
    sys.exit(1)

print("\n" + "="*80)
print("✓ ALL TESTS PASSED!")
print("  - Network architectures are compatible")
print("  - Parameters can be transferred")
print("  - Transferred network produces identical outputs")
print("="*80 + "\n")
