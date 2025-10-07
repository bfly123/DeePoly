#!/usr/bin/env python3
"""测试 General PDE Solver - Allen-Cahn 稳态方程"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.problem_solvers.general_pde_solver.solver import GeneralPDESolver
import numpy as np

if __name__ == "__main__":
    print("=" * 80)
    print("测试 General PDE Solver - Allen-Cahn 稳态方程")
    print("=" * 80)

    # 创建求解器
    case_dir = "cases/general_pde_cases/AC_equation"
    print(f"\n加载案例: {case_dir}")

    try:
        solver = GeneralPDESolver(case_dir=case_dir)
        print("✓ 求解器创建成功")
    except Exception as e:
        print(f"✗ 求解器创建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 运行求解
    print("\n开始求解...")
    try:
        results = solver.solve()
        print("\n✓ 求解完成")
    except Exception as e:
        print(f"\n✗ 求解失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 验证结果
    print("\n" + "=" * 80)
    print("验证结果")
    print("=" * 80)

    conv_info = results["convergence_info"]
    U_steady = results["solution"]

    if conv_info["converged"]:
        print("✓ 伪时间步收敛")
        print(f"  迭代次数: {conv_info['iterations']}")
        print(f"  最终残差: {conv_info['final_residual']:.6e}")
        print(f"  最终伪时间: {conv_info['final_tau']:.6f}")
        print(f"  总用时: {conv_info['elapsed_time']:.2f}s")
    else:
        print(f"✗ 未收敛: {conv_info.get('reason', 'unknown')}")
        print(f"  迭代次数: {conv_info['iterations']}")
        print(f"  最终残差: {conv_info['final_residual']:.6e}")
        sys.exit(1)

    # 分析解
    print("\n解的统计信息:")
    print(f"  解的范数: ||U|| = {np.linalg.norm(U_steady):.6e}")
    print(f"  解的范围: [{U_steady.min():.6f}, {U_steady.max():.6f}]")
    print(f"  解的均值: {U_steady.mean():.6f}")
    print(f"  解的标准差: {U_steady.std():.6f}")

    # 检查收敛历史
    if "residual_history" in conv_info:
        res_hist = np.array(conv_info["residual_history"])
        print(f"\n残差历史:")
        print(f"  初始残差: {res_hist[0]:.6e}")
        print(f"  最终残差: {res_hist[-1]:.6e}")
        print(f"  下降比例: {(1 - res_hist[-1]/res_hist[0])*100:.2f}%")

    print("\n" + "=" * 80)
    print("✓ 测试完成!")
    print("=" * 80)
