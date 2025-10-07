"""
简化测试：直接测试 fitter 的算子初始化
跳过Spotter训练，直接初始化模型和fitter
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from src.problem_solvers.general_pde_solver.core.fitter import GeneralPDEFitter
from src.problem_solvers.general_pde_solver.core.net import GeneralPDENet
from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig
from src.problem_solvers.general_pde_solver.utils.data import GeneralPDEDataGenerator

print("=" * 80)
print("简化测试：Fitter算子初始化")
print("=" * 80)

# 1. 加载配置
case_dir = "cases/general_pde_cases/AC_equation"
config = GeneralPDEConfig(case_dir=case_dir)
print(f"\n✓ 配置加载完成")
print(f"  问题类型: {config.problem_type}")
print(f"  维度: {config.n_dim}")
print(f"  方程数: {config.n_eqs}")
print(f"  Segments: {config.n_segments}")

# 2. 生成数据
datagen = GeneralPDEDataGenerator(config)
data_train = datagen.generate_data("train")
print(f"\n✓ 训练数据生成完成")
print(f"  训练点数: {len(data_train['x'])}")
print(f"  段数: {len(data_train['x_segments'])}")

# 3. 创建神经网络模型
model = GeneralPDENet(config).to(config.device)
print(f"\n✓ 神经网络模型创建完成")
n_params = sum(p.numel() for p in model.parameters())
print(f"  模型参数数量: {n_params}")

# 4. 创建fitter
fitter = GeneralPDEFitter(config=config, data=data_train, time_scheme='implicit_euler')
print(f"\n✓ Fitter创建完成")

# 5. 第一次初始化算子（使用初始模型）
print("\n" + "=" * 80)
print("第一次 fitter_init (初始模型)")
print("=" * 80)
fitter.fitter_init(model=model)

print(f"\n预编译状态: {fitter._precompiled}")
print(f"Segments数量: {fitter.ns}")

# 检查L1算子
seg_idx = 0
print(f"\n【Segment {seg_idx} 的算子】")
L1_ops = fitter._linear_operators[seg_idx].get("L1", None)
L2_ops = fitter._linear_operators[seg_idx].get("L2", None)

if L1_ops is not None:
    print(f"  L1算子数量: {len(L1_ops)}")
    for eq_idx, L1_mat in enumerate(L1_ops):
        is_zero = np.allclose(L1_mat, 0, atol=1e-10)
        norm = np.linalg.norm(L1_mat)
        print(f"    L1[{eq_idx}]: shape={L1_mat.shape}, norm={norm:.6e}, is_zero={is_zero}")

# 检查features
features = fitter._features[seg_idx]
print(f"\n  Features:")
print(f"    导数阶数: {len(features)}")
for deriv_idx, feat_mat in enumerate(features):
    print(f"    导数[{deriv_idx}]: shape={feat_mat.shape}")

print(f"\n  多项式自由度 (dg): {fitter.dg}")
print(f"  总自由度 (dgN): {fitter.dgN}")
print(f"  是否hybrid: {fitter.dgN > fitter.dg}")

# 6. 模拟训练：稍微改变模型参数
print("\n" + "=" * 80)
print("模拟训练：改变模型参数")
print("=" * 80)
for param in model.parameters():
    param.data += torch.randn_like(param) * 0.01
print("✓ 模型参数已随机扰动")

# 7. 第二次初始化算子（使用更新后的模型）
print("\n" + "=" * 80)
print("第二次 fitter_init (训练后模型)")
print("=" * 80)
fitter.fitter_init(model=model)

# 再次检查L1算子
L1_ops_new = fitter._linear_operators[seg_idx].get("L1", None)
if L1_ops_new is not None and L1_ops is not None:
    print(f"\n  L1算子对比:")
    for eq_idx in range(len(L1_ops)):
        diff = np.linalg.norm(L1_ops_new[eq_idx] - L1_ops[eq_idx])
        print(f"    L1[{eq_idx}] 差异: {diff:.6e}")
        print(f"      -> {'变化了' if diff > 1e-10 else '未变化'} (hybrid方法应该变化)")

# 8. 测试N算子
print("\n" + "=" * 80)
print("测试 N 算子")
print("=" * 80)

# 创建测试点
U_test = np.array([[0.0], [1.0], [0.5], [-1.0]])
features_test = fitter._features[seg_idx][0][:4]  # 前4个点

N_result = fitter.N_func(features_test, U_test)
print(f"\n  测试点: u = {U_test.flatten()}")
print(f"  N(u) = -5u + 5u³")

if isinstance(N_result, list):
    for eq_idx, N_vals in enumerate(N_result):
        expected = -5 * U_test.flatten() + 5 * U_test.flatten()**3
        print(f"\n  方程 {eq_idx}:")
        print(f"    预期: {expected}")
        print(f"    实际: {np.array(N_vals)}")
        error = np.abs(np.array(N_vals) - expected)
        print(f"    误差: {error}")
        print(f"    {'✓ 正确' if np.max(error) < 1e-6 else '✗ 错误'}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
