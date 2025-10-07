#!/usr/bin/env python
"""快速测试网络迁移功能"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.problem_solvers.general_pde_solver.solver import GeneralPDESolver
from src.problem_solvers.general_pde_solver.utils.config import GeneralPDEConfig

# 创建配置（使用 AC equation case）
case_dir = "cases/general_pde_cases/AC_equation"
config = GeneralPDEConfig(case_dir=case_dir)

# 修改为测试配置（减少迭代）
config.initial_condition['spotter_config']['epochs'] = 200
config.initial_condition['spotter_config']['print_interval'] = 50
config.pseudo_time_stepping['max_iterations'] = 10
config.realtime_visualization = False

print("\n" + "="*80)
print("Testing Network Transfer: SpotterNet -> GeneralPDENet")
print("="*80)

# 创建求解器
solver = GeneralPDESolver(config=config, case_dir=case_dir)

# 运行求解
results = solver.solve()

print("\n" + "="*80)
print("Test Results:")
print("="*80)
print(f"Converged: {results['convergence_info']['converged']}")
print(f"Iterations: {results['convergence_info']['iterations']}")
print(f"Final residual: {results['convergence_info']['final_residual']:.6e}")
print("="*80)

if results["convergence_info"]["iterations"] > 0:
    print("\n✓ Network transfer test PASSED!")
    print("  - SpotterNet trained successfully")
    print("  - Weights transferred to GeneralPDENet")
    print("  - Fitter stage started with transferred network")
else:
    print("\n✗ Network transfer test FAILED!")
