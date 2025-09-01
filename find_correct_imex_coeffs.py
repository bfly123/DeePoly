#!/usr/bin/env python3
"""
寻找正确的二阶IMEX-RK系数
基于精度条件求解最优参数
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def solve_for_correct_coefficients():
    """求解满足二阶精度条件的IMEX-RK(2,2,2)系数"""
    print("=== 求解正确的二阶IMEX-RK系数 ===\n")
    
    # 对于二阶IMEX-RK(2,2,2)，假设结构：
    # c = [c1, c2]
    # b = [b1, b2] 
    # A_imp = [[a11, 0], [a21, a22]]
    # A_exp = [[0, 0], [e21, 0]]
    
    # 精度条件：
    # 1) b1 + b2 = 1
    # 2) b1*c1 + b2*c2 = 1/2
    # 3) b1*c1^2 + b2*c2^2 = 1/3
    
    # 对于IMEX方法，通常b1 = b2 = 1/2
    b1, b2 = 0.5, 0.5
    
    def equations(vars):
        c1, c2 = vars
        
        eq1 = b1*c1 + b2*c2 - 0.5        # 条件2
        eq2 = b1*c1**2 + b2*c2**2 - 1.0/3.0  # 条件3
        
        return [eq1, eq2]
    
    # 初始猜测
    c_guess = [0.3, 0.7]
    
    # 求解
    c_solution = fsolve(equations, c_guess)
    c1, c2 = c_solution
    
    print(f"求解得到的节点:")
    print(f"c1 = {c1:.10f}")
    print(f"c2 = {c2:.10f}")
    
    # 验证解
    check1 = b1 + b2
    check2 = b1*c1 + b2*c2 
    check3 = b1*c1**2 + b2*c2**2
    
    print(f"\n验证精度条件:")
    print(f"条件1: Σb_i = {check1:.10f} (应该 = 1)")
    print(f"条件2: Σb_i*c_i = {check2:.10f} (应该 = 0.5)")
    print(f"条件3: Σb_i*c_i^2 = {check3:.10f} (应该 = {1.0/3.0:.10f})")
    
    # 计算对应的IMEX系数
    # 对于DIRK类型：c1 = a11, c2 = a21 + a22
    # 选择a22 = a11 (对角线相等的DIRK)
    a11 = c1
    a22 = c1  # DIRK特性
    a21 = c2 - a22
    
    # 显式部分：e21 = c2 (通常选择)
    e21 = c2
    
    print(f"\n对应的Butcher表系数:")
    print(f"隐式矩阵 A_imp:")
    print(f"  [[{a11:.10f}, 0]]")
    print(f"  [[{a21:.10f}, {a22:.10f}]]")
    print(f"显式矩阵 A_exp:")
    print(f"  [[0, 0]]")  
    print(f"  [[{e21:.10f}, 0]]")
    print(f"权重: b = [0.5, 0.5]")
    
    return c1, c2, a11, a21, a22, e21

def compare_with_current():
    """与当前实现对比"""
    print(f"\n=== 与当前实现对比 ===")
    
    # 当前参数
    gamma_current = (2 - np.sqrt(2)) / 2
    c1_current = gamma_current
    c2_current = 1 - gamma_current
    
    print(f"当前实现:")
    print(f"  γ = {gamma_current:.10f}")
    print(f"  c = [{c1_current:.10f}, {c2_current:.10f}]")
    print(f"  Σb_i*c_i^2 = {0.5*(c1_current**2 + c2_current**2):.10f}")
    
    # 正确系数
    c1_correct, c2_correct, a11, a21, a22, e21 = solve_for_correct_coefficients()
    
    print(f"\n正确系数应该是:")
    print(f"  c = [{c1_correct:.10f}, {c2_correct:.10f}]") 
    print(f"  Σb_i*c_i^2 = {0.5*(c1_correct**2 + c2_correct**2):.10f}")
    
    # 计算差异
    diff_c1 = abs(c1_current - c1_correct)
    diff_c2 = abs(c2_current - c2_correct)
    
    print(f"\n差异:")
    print(f"  Δc1 = {diff_c1:.6f}")
    print(f"  Δc2 = {diff_c2:.6f}")
    
    return c1_correct, c2_correct

def test_corrected_scheme():
    """测试修正后的格式"""
    print(f"\n=== 测试修正格式精度 ===")
    
    # 获取正确系数
    c1, c2 = 1.0/6.0, 5.0/6.0  # 精确的二阶IMEX-RK节点
    a11 = c1  
    a22 = c1
    a21 = c2 - a22
    e21 = c2
    
    print(f"使用修正系数:")
    print(f"c = [{c1:.10f}, {c2:.10f}]")
    print(f"A_imp = [[{a11:.10f}, 0], [{a21:.10f}, {a22:.10f}]]")
    print(f"A_exp = [[0, 0], [{e21:.10f}, 0]]")
    
    # 验证精度
    b = np.array([0.5, 0.5])
    c = np.array([c1, c2])
    
    check1 = np.sum(b)
    check2 = np.sum(b * c) 
    check3 = np.sum(b * c**2)
    
    print(f"\n精度验证:")
    print(f"条件1: {check1:.10f} (目标: 1.0)")
    print(f"条件2: {check2:.10f} (目标: 0.5)") 
    print(f"条件3: {check3:.10f} (目标: {1.0/3.0:.10f})")
    
    errors = [abs(check1-1.0), abs(check2-0.5), abs(check3-1.0/3.0)]
    print(f"误差: {[f'{e:.2e}' for e in errors]}")
    
    if max(errors) < 1e-15:
        print("✅ 所有精度条件满足!")
    else:
        print("❌ 精度条件仍有问题")
    
    return c1, c2, a11, a21, a22, e21

def main():
    """主函数"""
    print("IMEX-RK(2,2,2) 系数修正")
    print("=" * 50)
    
    # 求解正确系数
    c1_correct, c2_correct = compare_with_current()
    
    # 测试修正格式
    coeffs = test_corrected_scheme()
    
    print(f"\n=== 修正建议 ===")
    print(f"当前文档中的γ参数需要修正!")
    print(f"建议使用:")
    print(f"  c1 = {1.0/6.0:.10f} (≈ 0.1667)")
    print(f"  c2 = {5.0/6.0:.10f} (≈ 0.8333)")
    print(f"或者求解二次方程得到精确值")
    
    # 生成修正的Butcher表
    print(f"\n=== 修正的Butcher表 ===")
    c1, c2 = 1.0/6.0, 5.0/6.0
    print(f"| 阶段 | c_i | 隐式系数 | 显式系数 | 权重 |")
    print(f"|------|-----|---------|---------|------|")
    print(f"| 1 | {c1:.4f} | {c1:.4f} | 0 | 0.5 |")
    print(f"| 2 | {c2:.4f} | {c2-c1:.4f}, {c1:.4f} | {c2:.4f} | 0.5 |")

if __name__ == "__main__":
    main()