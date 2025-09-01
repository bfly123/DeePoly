#!/usr/bin/env python3
"""
Debug IMEX-RK(2,2,2) precision issues
分析当前实现的精度问题
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_butcher_coefficients():
    """分析当前Butcher表系数的精度条件"""
    print("=== IMEX-RK(2,2,2) 精度分析 ===\n")
    
    # 当前参数
    gamma = (2 - np.sqrt(2)) / 2
    print(f"当前 γ = {gamma:.10f}")
    
    # 当前Butcher表
    A_imp = np.array([[gamma, 0], [1 - 2*gamma, gamma]])
    A_exp = np.array([[0, 0], [1 - gamma, 0]])
    b = np.array([0.5, 0.5])
    c = np.array([gamma, 1 - gamma])  # 从A矩阵行和计算
    
    print(f"隐式矩阵 A_imp:\n{A_imp}")
    print(f"显式矩阵 A_exp:\n{A_exp}")
    print(f"权重向量 b: {b}")
    print(f"节点向量 c: {c}")
    
    # 验证二阶精度条件
    print("\n=== 二阶精度条件验证 ===")
    
    # 条件1: Σb_i = 1
    sum_b = np.sum(b)
    print(f"条件1 - Σb_i = {sum_b:.10f} (应该 = 1)")
    
    # 条件2: Σb_i*c_i = 1/2
    sum_bc = np.sum(b * c)
    print(f"条件2 - Σb_i*c_i = {sum_bc:.10f} (应该 = 0.5)")
    
    # 条件3: Σb_i*c_i^2 = 1/3
    sum_bc2 = np.sum(b * c**2)
    print(f"条件3 - Σb_i*c_i^2 = {sum_bc2:.10f} (应该 = 0.3333...)")
    
    # 计算误差
    error1 = abs(sum_b - 1.0)
    error2 = abs(sum_bc - 0.5)
    error3 = abs(sum_bc2 - 1.0/3.0)
    
    print(f"\n精度条件误差:")
    print(f"误差1: {error1:.2e}")
    print(f"误差2: {error2:.2e}")
    print(f"误差3: {error3:.2e}")
    
    if error3 > 1e-10:
        print(f"⚠️  条件3不满足！这可能导致精度下降到一阶")
        print(f"   理论值: {1.0/3.0:.10f}")
        print(f"   实际值: {sum_bc2:.10f}")
        print(f"   差值: {sum_bc2 - 1.0/3.0:.2e}")
    
    return gamma, A_imp, A_exp, b, c

def compare_l2f_implementations():
    """比较L2⊙F项的不同实现方式"""
    print("\n=== L2⊙F项实现对比 ===")
    
    # 模拟参数
    n_points = 5
    dgN = 3
    
    # 生成测试数据
    np.random.seed(42)
    L2 = np.random.randn(n_points, dgN)
    beta = np.random.randn(dgN)
    F_vals = np.random.randn(n_points) + 1  # 避免零值
    
    # 方法1: 原来的逐点相乘 (L2@β) * F
    L2_beta = L2 @ beta  # (n_points,)
    method1 = L2_beta * F_vals  # (n_points,)
    
    # 方法2: 当前的矩阵乘法 diag(F) @ L2 @ β
    method2 = np.diag(F_vals) @ L2 @ beta  # (n_points,)
    
    # 比较结果
    diff = np.linalg.norm(method1 - method2)
    print(f"L2@β的形状: {L2_beta.shape}")
    print(f"F值的形状: {F_vals.shape}")
    print(f"方法1结果: {method1}")
    print(f"方法2结果: {method2}")
    print(f"两种方法的差异: {diff:.2e}")
    
    if diff < 1e-14:
        print("✅ 两种L2⊙F实现方式数学等价")
    else:
        print("❌ 两种L2⊙F实现方式存在差异！")
    
    return method1, method2, diff

def test_simple_ode():
    """测试简单ODE的精度"""
    print("\n=== 简单ODE精度测试 ===")
    
    # 测试方程: du/dt = -u (解析解: u(t) = u0*exp(-t))
    def exact_solution(t, u0=1.0):
        return u0 * np.exp(-t)
    
    # IMEX-RK(2,2,2)参数
    gamma = (2 - np.sqrt(2)) / 2
    b = np.array([0.5, 0.5])
    
    # 时间参数
    t_final = 0.1
    dt_values = [0.1, 0.05, 0.025, 0.0125]
    errors = []
    
    u0 = 1.0
    
    for dt in dt_values:
        n_steps = int(t_final / dt)
        u = u0
        
        for step in range(n_steps):
            t_n = step * dt
            
            # 阶段1: [1 - γΔt*(-1)] β₁ = u + 0 (没有显式项)
            # 简化为: [1 + γΔt] β₁ = u
            beta1 = u / (1 + gamma * dt)
            u1 = beta1  # V = [1] for scalar case
            
            # 阶段2: [1 + γΔt] β₂ = u + (1-2γ)Δt*(-1)*β₁
            rhs2 = u - (1 - 2*gamma) * dt * beta1
            beta2 = rhs2 / (1 + gamma * dt)
            u2 = beta2
            
            # 最终更新: u^{n+1} = u + Δt * Σb_i * (-1) * β_i
            u_new = u + dt * (b[0] * (-beta1) + b[1] * (-beta2))
            u = u_new
        
        # 计算误差
        u_exact = exact_solution(t_final, u0)
        error = abs(u - u_exact)
        errors.append(error)
        
        print(f"dt = {dt:.4f}, u_numerical = {u:.8f}, u_exact = {u_exact:.8f}, error = {error:.2e}")
    
    # 分析收敛阶数
    print(f"\n=== 收敛阶数分析 ===")
    for i in range(1, len(errors)):
        ratio = errors[i-1] / errors[i]
        order = np.log2(ratio)
        print(f"dt: {dt_values[i-1]:.4f} -> {dt_values[i]:.4f}, 误差比: {ratio:.3f}, 阶数: {order:.3f}")
    
    return dt_values, errors

def main():
    """主函数"""
    print("IMEX-RK(2,2,2) 精度问题诊断")
    print("=" * 50)
    
    # 分析Butcher表系数
    gamma, A_imp, A_exp, b, c = analyze_butcher_coefficients()
    
    # 比较L2⊙F实现
    method1, method2, diff = compare_l2f_implementations()
    
    # 测试简单ODE
    dt_values, errors = test_simple_ode()
    
    # 绘制收敛性图
    plt.figure(figsize=(10, 6))
    plt.loglog(dt_values, errors, 'bo-', label='IMEX-RK(2,2,2)')
    plt.loglog(dt_values, [err * (dt_values[0]/dt)**1 for dt, err in zip(dt_values, errors)], 'r--', label='一阶收敛')
    plt.loglog(dt_values, [errors[0] * (dt_values[0]/dt)**2 for dt in dt_values], 'g--', label='二阶收敛')
    plt.xlabel('时间步长 dt')
    plt.ylabel('误差')
    plt.title('IMEX-RK(2,2,2) 收敛性测试')
    plt.legend()
    plt.grid(True)
    plt.savefig('imex_convergence_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== 总结 ===")
    print(f"1. Butcher表系数满足基本精度条件: 需要检查条件3")
    print(f"2. L2⊙F实现方式数学等价性: {'是' if diff < 1e-14 else '否'}")
    print(f"3. 简单ODE测试收敛性: 查看图像和阶数分析")

if __name__ == "__main__":
    main()