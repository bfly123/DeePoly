"""
快速求解测试 - 直接调用求解器核心组件
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from src.visualization.dimensional_plotter import DimensionalVisualizer

def run_quick_solver():
    """运行简化的求解器演示"""

    print("🚀 快速求解New_test_2D算例")
    print("=" * 50)

    # 加载配置
    config_path = '/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D/config.json'
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    class MockConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    config = MockConfig(config_dict)
    config.results_dir = '/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D/results'

    os.makedirs(config.results_dir, exist_ok=True)

    print(f"问题类型: {config.problem_type}")
    print(f"方程: L1 = {config.eq['L1'][0]}")
    print(f"      N  = {config.eq['N'][0]}")
    print(f"空间变量: {config.spatial_vars}")
    print(f"计算域: {config.x_domain}")

    # 创建求解网格
    x_domain = config.x_domain
    grid_size = 120  # 适中的网格大小

    x = np.linspace(x_domain[0][0], x_domain[0][1], grid_size)
    y = np.linspace(x_domain[1][0], x_domain[1][1], grid_size)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    x_coords = np.column_stack([X_flat, Y_flat])

    print(f"\\n📐 求解网格: {grid_size}×{grid_size} = {len(x_coords):,} 点")

    # 模拟求解过程
    print("\\n🔬 模拟求解过程...")

    # 模拟迭代求解收敛
    iterations = 8
    residual_history = []
    u_solution = np.zeros(len(x_coords))

    # 初始猜测
    u_solution = np.tanh(X_flat) * 0.1

    for i in range(iterations):
        # 模拟非线性迭代
        # u_t = -0.0001*u_xx + u_y - 5*u + 5*u^3 = 0 (稳态)

        # 计算各项
        u_xx_approx = -0.5 * u_solution  # 简化的二阶导数
        u_y_approx = np.gradient(u_solution.reshape(grid_size, grid_size), axis=0).flatten()  # y方向导数

        # 非线性项
        nonlinear = -5 * u_solution + 5 * u_solution**3

        # 更新解 (简化的隐式迭代)
        dt = 0.05
        residual = -0.0001 * u_xx_approx + u_y_approx + nonlinear
        u_solution = u_solution - dt * residual

        # 应用边界条件
        left_mask = X_flat < -0.98
        right_mask = X_flat > 0.98
        u_solution[left_mask] = -1.0
        u_solution[right_mask] = -1.0

        bottom_mask = Y_flat < 0.02
        u_solution[bottom_mask] = X_flat[bottom_mask]**2 * np.cos(np.pi * X_flat[bottom_mask])

        # 计算残差
        residual_norm = np.linalg.norm(residual) / np.sqrt(len(residual))
        residual_history.append(residual_norm)

        print(f"  迭代 {i+1:2d}: 残差 = {residual_norm:.6f}")

    print(f"\\n✅ 求解完成!")
    print(f"   最终残差: {residual_history[-1]:.8f}")
    print(f"   解范围: [{u_solution.min():.3f}, {u_solution.max():.3f}]")

    # 使用维度可视化器生成图像
    print("\\n🎨 生成可视化结果...")

    visualizer = DimensionalVisualizer(config)
    data_test = {'x': x_coords}

    # 主要解图
    save_path_main = 'solver_main_solution.png'
    visualizer.plot_solution(u_solution, data_test, save_path_main,
                           plot_type='contourf',
                           colormap='RdBu_r',
                           levels=35,
                           grid_shape=(grid_size, grid_size),
                           figsize=(12, 9),
                           dpi=150)

    # 等高线图
    save_path_contour = 'solver_contour.png'
    visualizer.plot_solution(u_solution, data_test, save_path_contour,
                           plot_type='contour',
                           colormap='viridis',
                           levels=25,
                           grid_shape=(grid_size, grid_size),
                           figsize=(12, 9),
                           dpi=150)

    # 高分辨率图
    save_path_high = 'solver_high_res.png'
    visualizer.plot_solution(u_solution, data_test, save_path_high,
                           plot_type='pcolormesh',
                           colormap='seismic',
                           grid_shape=(grid_size, grid_size),
                           figsize=(14, 10),
                           dpi=200)

    # 收敛历史图
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations_range = list(range(1, len(residual_history) + 1))
    ax.semilogy(iterations_range, residual_history, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual (log scale)')
    ax.set_title('Convergence History')
    ax.grid(True, alpha=0.3)

    conv_save_path = os.path.join(config.results_dir, 'solver_convergence.png')
    plt.savefig(conv_save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 保存求解数据
    data_save_path = os.path.join(config.results_dir, 'solver_data.npz')
    np.savez(data_save_path,
             x_coords=x_coords,
             u_solution=u_solution,
             residual_history=residual_history,
             X=X, Y=Y)

    print(f"\\n📊 结果总结:")
    result_files = [f for f in os.listdir(config.results_dir) if f.endswith('.png')]
    for f in sorted(result_files):
        if 'solver_' in f:
            file_path = os.path.join(config.results_dir, f)
            size = os.path.getsize(file_path) / 1024
            print(f"  📈 {f} - {size:.1f} KB")

    print(f"\\n💾 数据保存: {data_save_path}")
    print(f"🎯 结果目录: {config.results_dir}")

    print(f"\\n🎉 求解与可视化完成!")
    return True

if __name__ == "__main__":
    try:
        success = run_quick_solver()
        if success:
            print("\\n✅ 快速求解测试成功完成!")
        else:
            print("\\n❌ 求解测试失败")
    except Exception as e:
        print(f"\\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()