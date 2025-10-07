#!/usr/bin/env python3
"""测试 compute_residual() 功能"""

import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.problem_solvers.general_pde_solver.solver import GeneralPDESolver

if __name__ == "__main__":
    print("=" * 80)
    print("测试 compute_residual() 功能")
    print("=" * 80)

    # 创建求解器
    case_dir = "cases/general_pde_cases/AC_equation"
    solver = GeneralPDESolver(case_dir=case_dir)

    print("\n运行Spotter阶段...")
    U_init = solver._spotter_stage()

    print("\n初始化Fitter...")
    solver.fitter.fitter_init(model=solver.model)

    print("\n运行5次伪时间步迭代...")
    # 手动运行几次迭代以设置_coeffs_current
    U_seg = solver.fitter.global_to_segments(U_init)
    coeffs = solver.fitter._initialize_coefficients()
    solver.fitter._coeffs_current = coeffs

    dt = 0.01
    for i in range(5):
        U_new, U_seg_new, coeffs_new = solver.fitter.time_step(
            U_init, U_seg, dt,
            coeffs_n=coeffs,
            current_time=0.0,
            step=i
        )
        U_init = U_new
        U_seg = U_seg_new
        coeffs = coeffs_new
        solver.fitter._coeffs_current = coeffs
        print(f"  迭代 {i+1}: ||U|| = {np.linalg.norm(U_new):.6e}")

    print("\n测试 compute_residual()...")
    try:
        residual = solver.fitter.compute_residual(U_new)

        print("✓ compute_residual() 成功执行")
        print(f"\n残差统计:")
        print(f"  ||R|| = {np.linalg.norm(residual):.6e}")
        print(f"  max|R| = {np.max(np.abs(residual)):.6e}")
        print(f"  mean|R| = {np.mean(np.abs(residual)):.6e}")
        print(f"  残差形状: {residual.shape}")

        # 检查残差的组成
        print(f"\n残差分解 (第一个点):")
        print(f"  R[0] = {residual[0]}")

        print("\n" + "=" * 80)
        print("✓ 测试完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ compute_residual() 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
