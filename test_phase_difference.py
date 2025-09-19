#!/usr/bin/env python3
"""
测试KDV方程数值解和参考解的相位差问题
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import scipy.io as sio
from scipy.interpolate import interp1d
import json
import sys
import os

def load_reference_solution():
    """加载参考解"""
    mat_data = sio.loadmat('cases/Time_pde_cases/KDV_equation/reference_data/KdV_cos.mat')
    t_ref = mat_data['t'].flatten()
    x_ref = mat_data['x'].flatten()
    usol_ref = mat_data['usol'].T  # (time, space)

    # 创建插值函数
    interpolators = {}
    for i in range(len(x_ref)):
        interpolators[i] = interp1d(t_ref, usol_ref[:, i], kind='cubic',
                                  bounds_error=False, fill_value='extrapolate')

    return {
        't_ref': t_ref,
        'x_ref': x_ref,
        'usol_ref': usol_ref,
        'interpolators': interpolators
    }

def get_reference_at_time(ref_data, t):
    """获取指定时间的参考解"""
    x_ref = ref_data['x_ref']
    interpolators = ref_data['interpolators']

    u_at_t = np.zeros(len(x_ref))
    for i in range(len(x_ref)):
        u_at_t[i] = interpolators[i](t)

    return u_at_t

def analyze_time_steps():
    """分析前几个时间步的解"""
    print("=== KDV方程相位差分析 ===")

    # 加载参考解
    ref_data = load_reference_solution()
    t_ref = ref_data['t_ref']
    x_ref = ref_data['x_ref']

    # 测试时间点
    test_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    print(f"参考解信息:")
    print(f"  时间范围: [{t_ref[0]:.2f}, {t_ref[-1]:.2f}]")
    print(f"  时间步长: {t_ref[1] - t_ref[0]:.2f}")
    print(f"  空间点数: {len(x_ref)}")
    print(f"  空间范围: [{x_ref[0]:.2f}, {x_ref[-1]:.2f}]")

    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, t in enumerate(test_times):
        if i >= len(axes):
            break

        # 获取参考解
        u_ref = get_reference_at_time(ref_data, t)

        # 绘制
        axes[i].plot(x_ref, u_ref, 'b-', linewidth=2, label=f'Reference t={t:.1f}')
        axes[i].set_title(f'Solution at t={t:.1f}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('u')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        axes[i].set_xlim(-20, 20)

        # 分析解的特征
        max_val = np.max(u_ref)
        min_val = np.min(u_ref)
        max_pos = x_ref[np.argmax(u_ref)]
        min_pos = x_ref[np.argmin(u_ref)]

        print(f"\nt={t:.1f}:")
        print(f"  最大值: {max_val:.6f} at x={max_pos:.3f}")
        print(f"  最小值: {min_val:.6f} at x={min_pos:.3f}")
        print(f"  解的范围: [{min_val:.6f}, {max_val:.6f}]")

        # 检查波形传播
        if t > 0:
            # 计算相对于初始条件的变化
            u_initial = get_reference_at_time(ref_data, 0.0)
            change = np.max(np.abs(u_ref - u_initial))
            print(f"  相对初始条件的最大变化: {change:.6f}")

            # 检查峰值移动
            initial_max_pos = x_ref[np.argmax(u_initial)]
            peak_shift = max_pos - initial_max_pos
            print(f"  峰值位移: {peak_shift:.6f}")

    plt.tight_layout()
    plt.savefig('cases/Time_pde_cases/KDV_equation/reference_time_evolution.png',
               dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n参考解时间演化图已保存到: cases/Time_pde_cases/KDV_equation/reference_time_evolution.png")

    # 分析时间导数来检查演化速度
    print(f"\n=== 时间导数分析 ===")
    dt_ref = t_ref[1] - t_ref[0]
    for i, t in enumerate([0.0, 0.2, 0.4]):
        u_curr = get_reference_at_time(ref_data, t)
        u_next = get_reference_at_time(ref_data, t + dt_ref)
        dudt = (u_next - u_curr) / dt_ref

        max_dudt = np.max(np.abs(dudt))
        print(f"  t={t:.1f}: max|∂u/∂t| = {max_dudt:.6f}")

def check_boundary_conditions():
    """检查边界条件的周期性"""
    print(f"\n=== 边界条件检查 ===")

    ref_data = load_reference_solution()
    x_ref = ref_data['x_ref']

    test_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for t in test_times:
        u_ref = get_reference_at_time(ref_data, t)

        # 检查左右边界值
        u_left = u_ref[0]  # x = -20
        u_right = u_ref[-1]  # x ≈ 20

        boundary_diff = abs(u_left - u_right)
        print(f"  t={t:.1f}: u(-20)={u_left:.6f}, u(+20)={u_right:.6f}, 差异={boundary_diff:.6e}")

if __name__ == "__main__":
    analyze_time_steps()
    check_boundary_conditions()

    print(f"\n=== 总结 ===")
    print(f"如果数值解存在相位差，可能原因:")
    print(f"1. 时间积分方法的数值耗散/色散")
    print(f"2. 空间离散化方法的差异")
    print(f"3. 边界条件处理的细微差异")
    print(f"4. 非线性项处理方式的不同")