"""
测试算子初始化的正确性

验证每次调用 fitter_init 后：
1. L1算子矩阵是否正确编译
2. N算子函数是否正确
3. features是否包含神经网络部分
"""

import numpy as np
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.problem_solvers.general_pde_solver.solver import GeneralPDESolver
from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig

# Load config
case_dir = "cases/general_pde_cases/AC_equation"
config = GeneralPDEConfig(case_dir=case_dir)

# Create solver
solver = GeneralPDESolver(config=config, case_dir=case_dir)

print("=" * 80)
print("测试算子初始化")
print("=" * 80)

# Initialize fitter
solver.fitter.fitter_init(model=solver.model)

print("\n1. 检查预编译状态:")
print(f"   _precompiled: {solver.fitter._precompiled}")

print("\n2. 检查线性算子 (L1):")
for seg_idx in range(min(2, solver.fitter.ns)):
    L1_ops = solver.fitter._linear_operators[seg_idx].get("L1", None)
    if L1_ops is not None:
        print(f"   Segment {seg_idx}:")
        print(f"     L1算子数量: {len(L1_ops)}")
        for eq_idx, L1_matrix in enumerate(L1_ops):
            print(f"     L1[{eq_idx}] shape: {L1_matrix.shape}")
            print(f"     L1[{eq_idx}] norm: {np.linalg.norm(L1_matrix):.6e}")
            # 检查是否为零矩阵
            is_zero = np.allclose(L1_matrix, 0)
            print(f"     L1[{eq_idx}] 是否为零: {is_zero}")

print("\n3. 检查非线性算子 (N):")
# 测试N算子在一个随机点上的值
U_test = np.random.randn(10, config.n_eqs)
seg_idx = 0
features_test = solver.fitter._features[seg_idx][0]  # 零阶导数特征
N_result = solver.fitter.N_func(features_test, U_test)
print(f"   Segment {seg_idx}:")
print(f"   测试点数: {U_test.shape[0]}")
print(f"   N算子输出: {type(N_result)}")
if isinstance(N_result, list):
    print(f"   N算子方程数: {len(N_result)}")
    for eq_idx, N_eq in enumerate(N_result):
        print(f"     N[{eq_idx}] shape: {np.array(N_eq).shape}")
        print(f"     N[{eq_idx}] 样本值: {N_eq[:3]}")

print("\n4. 检查features构成:")
for seg_idx in range(min(2, solver.fitter.ns)):
    features = solver.fitter._features[seg_idx]
    print(f"   Segment {seg_idx}:")
    print(f"     导数阶数: {len(features)}")
    for deriv_idx, feature_matrix in enumerate(features):
        print(f"     导数[{deriv_idx}] shape: {feature_matrix.shape}")
        # 检查是否包含神经网络部分
        dgN = solver.fitter.dgN
        dg = solver.fitter.dg
        has_nn = (feature_matrix.shape[1] > dg)
        print(f"     导数[{deriv_idx}] 包含神经网络: {has_nn} (dgN={dgN}, dg={dg})")

print("\n5. 手动验证L1算子公式:")
# L1: diff(u,t) - 0.0001*diff(u,x,2)
# 应该是 features[dt] - 0.0001 * features[dxx]
seg_idx = 0
features_list = solver.fitter._features[seg_idx]
print(f"   可用导数特征索引: {len(features_list)}")
print(f"   all_derivatives: {solver.fitter.all_derivatives}")

# 检查L1算子是否正确组合了导数特征
L1_ops = solver.fitter._linear_operators[seg_idx].get("L1", None)
if L1_ops is not None and len(L1_ops) > 0:
    L1_matrix = L1_ops[0]
    print(f"   L1矩阵非零元素数: {np.count_nonzero(L1_matrix)}")
    print(f"   L1矩阵最大值: {np.max(np.abs(L1_matrix)):.6e}")

print("\n6. 测试N算子的正确性:")
# N = -5*u + 5*u**3
# 对于u=1, 应该得到 -5 + 5 = 0
# 对于u=0, 应该得到 0
U_test_simple = np.array([[0.0], [1.0], [2.0]])
N_result_simple = solver.fitter.N_func(features_test[:3], U_test_simple)
print(f"   测试点: u = {U_test_simple.flatten()}")
print(f"   N(u) 预期值: {-5*U_test_simple.flatten() + 5*U_test_simple.flatten()**3}")
print(f"   N(u) 实际值: {np.array(N_result_simple[0])}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
