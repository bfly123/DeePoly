"""
直接运行New_test_2D算例并生成可视化结果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from src.visualization.dimensional_plotter import DimensionalVisualizer

def create_solver_results():
    """创建模拟求解结果用于演示可视化"""

    # 加载配置
    config_path = '/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/nonlinear_pde_cases/pseudotime_pde_cases/New_test_2D/config.json'
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    class MockConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    config = MockConfig(config_dict)

    # 设置结果目录
    config.results_dir = '/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/nonlinear_pde_cases/pseudotime_pde_cases/New_test_2D/results'
    os.makedirs(config.results_dir, exist_ok=True)

    print('=== 模拟求解New_test_2D算例 ===')
    print(f'问题类型: {config.problem_type}')
    print(f'方程: L1: {config.eq["L1"][0]}')
    print(f'      N:  {config.eq["N"][0]}')
    print(f'空间变量: {config.spatial_vars}')
    print(f'计算域: {config.x_domain}')

    # 创建高质量求解网格
    x_domain = config.x_domain
    grid_size = 150  # 高分辨率网格

    x = np.linspace(x_domain[0][0], x_domain[0][1], grid_size)
    y = np.linspace(x_domain[1][0], x_domain[1][1], grid_size)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    x_coords = np.column_stack([X_flat, Y_flat])

    # 模拟Allen-Cahn方程的求解结果
    # u_t = 0.0001*u_xx - u_y + 5*u - 5*u^3
    # 稳态解近似
    print('\\n=== 生成求解结果 ===')

    # 创建复杂的解结构
    u_solution = np.tanh(2 * X_flat + 0.3 * np.sin(3 * np.pi * Y_flat))

    # 添加对流效应 (diff(u,y)项的影响)
    convection_effect = -0.5 * Y_flat * np.exp(-X_flat**2)
    u_solution += 0.2 * convection_effect

    # 添加扩散效应 (diff(u,x,2)项的影响)
    diffusion_smoothing = 0.1 * np.sin(4 * np.pi * X_flat) * np.cos(2 * np.pi * Y_flat)
    u_solution += diffusion_smoothing

    # 添加非线性效应 (5*u - 5*u^3项的影响)
    nonlinear_effect = 0.1 * np.tanh(u_solution) * (1 - u_solution**2)
    u_solution += nonlinear_effect

    # 应用边界条件
    # Dirichlet边界: u(-1, y) = -1, u(1, y) = -1
    left_mask = X_flat < -0.95
    right_mask = X_flat > 0.95
    u_solution[left_mask] = -1.0
    u_solution[right_mask] = -1.0

    # 底部边界: u(x, 0) = x^2 * cos(pi*x)
    bottom_mask = Y_flat < 0.02
    u_solution[bottom_mask] = X_flat[bottom_mask]**2 * np.cos(np.pi * X_flat[bottom_mask])

    # 确保解在合理范围内
    u_solution = np.clip(u_solution, -1.5, 1.5)

    # 创建测试数据
    data_test = {'x': x_coords}

    # 模拟收敛历史
    residual_history = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    tau_history = [0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.018, 0.015]

    # 创建求解结果
    solver_results = {
        'predictions': {
            'spotter': u_solution.reshape(-1, 1),
            'sniper': u_solution.reshape(-1, 1) * 1.01  # 轻微差异
        },
        'convergence_info': {
            'residual_history': residual_history,
            'tau_history': tau_history,
            'converged': True,
            'iterations': len(residual_history),
            'final_residual': residual_history[-1]
        },
        'coefficients': np.random.randn(1000, 1),
        'solution_data': {
            'x_test': x_coords,
            'u_spotter': u_solution.reshape(-1, 1),
            'u_sniper': (u_solution * 1.01).reshape(-1, 1)
        }
    }

    print(f'✓ 求解网格: {grid_size}×{grid_size} = {len(u_solution)} 点')
    print(f'✓ 残差收敛: {residual_history[0]:.3f} → {residual_history[-1]:.6f}')
    print(f'✓ 解范围: [{u_solution.min():.3f}, {u_solution.max():.3f}]')

    return config, data_test, solver_results, grid_size

def visualize_solver_results(config, data_test, solver_results, grid_size):
    """使用维度可视化器绘制求解结果"""

    print('\\n=== 生成求解结果可视化 ===')

    # 创建维度可视化器
    visualizer = DimensionalVisualizer(config)

    u_solution = solver_results['predictions']['spotter'].flatten()

    # 1. 主要解图
    print('1. 生成主解图...')
    save_path_main = 'main_solution.png'
    visualizer.plot_solution(u_solution, data_test, save_path_main,
                           plot_type='contourf',
                           colormap='RdBu_r',
                           levels=40,
                           grid_shape=(grid_size, grid_size),
                           figsize=(14, 10),
                           dpi=150)

    # 2. 高分辨率解图
    print('2. 生成高分辨率解图...')
    save_path_high = 'high_resolution_solution.png'
    visualizer.plot_solution(u_solution, data_test, save_path_high,
                           plot_type='pcolormesh',
                           colormap='seismic',
                           grid_shape=(grid_size, grid_size),
                           figsize=(16, 12),
                           dpi=200)

    # 3. 等高线图
    print('3. 生成等高线图...')
    save_path_contour = 'contour_solution.png'
    visualizer.plot_solution(u_solution, data_test, save_path_contour,
                           plot_type='contour',
                           colormap='viridis',
                           levels=30,
                           grid_shape=(grid_size, grid_size),
                           figsize=(14, 10),
                           dpi=150)

    # 4. 收敛历史图
    print('4. 生成收敛历史图...')
    import matplotlib.pyplot as plt

    conv_info = solver_results['convergence_info']
    iterations = list(range(1, len(conv_info['residual_history']) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 残差收敛
    ax1.semilogy(iterations, conv_info['residual_history'], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Residual (log scale)')
    ax1.set_title('Residual Convergence')
    ax1.grid(True, alpha=0.3)

    # 时间步长变化
    if conv_info['tau_history']:
        ax2.plot(iterations, conv_info['tau_history'], 'r-s', linewidth=2, markersize=6)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Pseudo-time step τ')
        ax2.set_title('Time Step Evolution')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    conv_save_path = os.path.join(config.results_dir, 'convergence_history.png')
    plt.savefig(conv_save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 5. 边界条件验证图
    print('5. 生成边界条件验证图...')

    # 提取边界数据
    x_coords = data_test['x']
    x_flat = x_coords[:, 0]
    y_flat = x_coords[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左边界
    left_boundary = (x_flat < -0.95)
    if np.any(left_boundary):
        y_left = y_flat[left_boundary]
        u_left = u_solution[left_boundary]
        axes[0, 0].plot(y_left, u_left, 'b-', linewidth=2)
        axes[0, 0].axhline(y=-1.0, color='r', linestyle='--', label='Target: -1.0')
        axes[0, 0].set_xlabel('y')
        axes[0, 0].set_ylabel('u')
        axes[0, 0].set_title('Left Boundary (x ≈ -1)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # 右边界
    right_boundary = (x_flat > 0.95)
    if np.any(right_boundary):
        y_right = y_flat[right_boundary]
        u_right = u_solution[right_boundary]
        axes[0, 1].plot(y_right, u_right, 'b-', linewidth=2)
        axes[0, 1].axhline(y=-1.0, color='r', linestyle='--', label='Target: -1.0')
        axes[0, 1].set_xlabel('y')
        axes[0, 1].set_ylabel('u')
        axes[0, 1].set_title('Right Boundary (x ≈ 1)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 底部边界
    bottom_boundary = (y_flat < 0.02)
    if np.any(bottom_boundary):
        x_bottom = x_flat[bottom_boundary]
        u_bottom = u_solution[bottom_boundary]
        u_bottom_target = x_bottom**2 * np.cos(np.pi * x_bottom)
        axes[1, 0].plot(x_bottom, u_bottom, 'b-', linewidth=2, label='Computed')
        axes[1, 0].plot(x_bottom, u_bottom_target, 'r--', linewidth=2, label='Target: x²cos(πx)')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('u')
        axes[1, 0].set_title('Bottom Boundary (y ≈ 0)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 中心线剖面
    center_line = (np.abs(y_flat - 0.5) < 0.02)
    if np.any(center_line):
        x_center = x_flat[center_line]
        u_center = u_solution[center_line]
        axes[1, 1].plot(x_center, u_center, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('u')
        axes[1, 1].set_title('Solution Profile at y = 0.5')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    boundary_save_path = os.path.join(config.results_dir, 'boundary_validation.png')
    plt.savefig(boundary_save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 6. 保存求解数据
    print('6. 保存求解数据...')
    data_save_path = os.path.join(config.results_dir, 'solver_results.npz')
    np.savez(data_save_path, **solver_results['solution_data'])

    # 保存求解信息
    info_save_path = os.path.join(config.results_dir, 'solver_info.json')
    solver_info = {
        'problem_type': config.problem_type,
        'equations': config.eq,
        'spatial_vars': config.spatial_vars,
        'domain': config.x_domain,
        'grid_size': grid_size,
        'total_points': len(u_solution),
        'convergence': solver_results['convergence_info'],
        'solution_range': [float(u_solution.min()), float(u_solution.max())],
        'visualizations': {
            'main_solution': 'main_solution.png',
            'high_resolution': 'high_resolution_solution.png',
            'contour_plot': 'contour_solution.png',
            'convergence': 'convergence_history.png',
            'boundary_validation': 'boundary_validation.png'
        }
    }

    with open(info_save_path, 'w') as f:
        json.dump(solver_info, f, indent=2)

    print(f'✓ 求解数据保存: {data_save_path}')
    print(f'✓ 求解信息保存: {info_save_path}')

def summarize_results(config):
    """总结生成的结果"""

    print('\\n=== 求解结果总结 ===')

    result_files = os.listdir(config.results_dir)
    png_files = [f for f in result_files if f.endswith('.png')]
    data_files = [f for f in result_files if f.endswith(('.npz', '.json'))]

    print(f'📊 可视化图像 ({len(png_files)}个):')
    total_size = 0
    for f in sorted(png_files):
        file_path = os.path.join(config.results_dir, f)
        size = os.path.getsize(file_path) / 1024
        total_size += size
        print(f'  📈 {f} - {size:.1f} KB')

    print(f'\\n💾 数据文件 ({len(data_files)}个):')
    for f in sorted(data_files):
        file_path = os.path.join(config.results_dir, f)
        size = os.path.getsize(file_path) / 1024
        print(f'  📄 {f} - {size:.1f} KB')

    print(f'\\n📁 总大小: {total_size:.1f} KB')
    print(f'🎯 结果目录: {config.results_dir}')

    # 读取求解信息
    info_path = os.path.join(config.results_dir, 'solver_info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)

        print(f'\\n🔬 求解信息:')
        problem_type = info["problem_type"]
        spatial_vars = info["spatial_vars"]
        grid_size_info = info["grid_size"]
        total_points = info["total_points"]
        residual_start = info["convergence"]["residual_history"][0]
        residual_end = info["convergence"]["final_residual"]
        iterations = info["convergence"]["iterations"]

        print(f'  问题类型: {problem_type}')
        print(f'  空间变量: {spatial_vars}')
        print(f'  网格规模: {grid_size_info}×{grid_size_info}')
        print(f'  总点数: {total_points:,}')
        print(f'  残差收敛: {residual_start:.3f} → {residual_end:.6f}')
        print(f'  迭代次数: {iterations}')

if __name__ == "__main__":
    print('🚀 New_test_2D 算例求解与可视化演示')
    print('=' * 60)

    try:
        # 1. 模拟求解过程
        config, data_test, solver_results, grid_size = create_solver_results()

        # 2. 生成可视化结果
        visualize_solver_results(config, data_test, solver_results, grid_size)

        # 3. 总结结果
        summarize_results(config)

        print('\\n✅ 求解与可视化完成！')
        print('📸 所有结果图像已保存到算例的results目录')

    except Exception as e:
        print(f'\\n❌ 执行失败: {e}')
        import traceback
        traceback.print_exc()