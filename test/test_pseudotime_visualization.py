"""
测试伪时间步算例的可视化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from src.visualization.dimensional_plotter import DimensionalVisualizer

def test_pseudotime_case():
    """测试伪时间步算例的可视化"""

    print("🚀 测试伪时间步算例New_test_2D可视化")
    print("=" * 50)

    # 加载配置
    config_path = '/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/nonlinear_pde_cases/pseudotime_pde_cases/New_test_2D/config.json'
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    class MockConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    config = MockConfig(config_dict)
    config.results_dir = '/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/nonlinear_pde_cases/pseudotime_pde_cases/New_test_2D/results'

    os.makedirs(config.results_dir, exist_ok=True)

    print(f"问题类型: {config.problem_type}")
    print(f"方程: L1 = {config.eq['L1'][0]}")
    print(f"      N  = {config.eq['N'][0]}")
    print(f"空间变量: {config.spatial_vars}")
    print(f"计算域: {config.x_domain}")
    print(f"伪时间步参数: max_iterations={config.pseudo_time.get('max_iterations', 'N/A')}")

    # 检查是否有现成的求解结果
    solver_data_path = os.path.join(config.results_dir, 'solver_results.npz')
    if os.path.exists(solver_data_path):
        print(f"\\n📁 加载现有求解数据: {solver_data_path}")
        solver_data = np.load(solver_data_path)
        x_coords = solver_data['x_test']
        u_solution = solver_data['u_spotter'].flatten()
        print(f"   坐标形状: {x_coords.shape}")
        print(f"   解形状: {u_solution.shape}")
        print(f"   解范围: [{u_solution.min():.3f}, {u_solution.max():.3f}]")
    else:
        # 创建新的求解数据
        print(f"\\n🔬 生成伪时间步求解数据...")

        x_domain = config.x_domain
        grid_size = 120
        x = np.linspace(x_domain[0][0], x_domain[0][1], grid_size)
        y = np.linspace(x_domain[1][0], x_domain[1][1], grid_size)
        X, Y = np.meshgrid(x, y)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        x_coords = np.column_stack([X_flat, Y_flat])

        # 模拟伪时间步求解的Allen-Cahn方程
        # u_t = 0.0001*u_xx - u_y + 5*u - 5*u^3 = 0 (稳态)

        # 初始条件：从边界条件初始化
        u_solution = np.zeros(len(x_coords))

        # 应用底部边界条件作为初始猜测
        bottom_mask = Y_flat < 0.02
        u_solution[bottom_mask] = X_flat[bottom_mask]**2 * np.cos(np.pi * X_flat[bottom_mask])

        # 伪时间步迭代
        dt = 0.02
        max_iter = 15
        residual_history = []

        print(f"   伪时间步迭代 (dt={dt}, max_iter={max_iter}):")

        for iter in range(max_iter):
            # 计算各项 (简化)
            u_xx_approx = np.gradient(np.gradient(u_solution.reshape(grid_size, grid_size), axis=1), axis=1).flatten()
            u_y_approx = np.gradient(u_solution.reshape(grid_size, grid_size), axis=0).flatten()

            # 非线性项
            nonlinear = 5 * u_solution - 5 * u_solution**3

            # 伪时间步更新 (向后欧拉)
            residual = 0.0001 * u_xx_approx - u_y_approx + nonlinear
            u_solution_new = u_solution + dt * residual

            # 应用边界条件
            left_mask = X_flat < -0.98
            right_mask = X_flat > 0.98
            u_solution_new[left_mask] = -1.0
            u_solution_new[right_mask] = -1.0

            bottom_mask = Y_flat < 0.02
            u_solution_new[bottom_mask] = X_flat[bottom_mask]**2 * np.cos(np.pi * X_flat[bottom_mask])

            # 计算残差
            residual_norm = np.linalg.norm(u_solution_new - u_solution) / np.sqrt(len(u_solution))
            residual_history.append(residual_norm)

            u_solution = u_solution_new

            print(f"     迭代 {iter+1:2d}: 变化 = {residual_norm:.6f}")

            if residual_norm < 1e-4:
                print(f"     收敛于迭代 {iter+1}")
                break

        print(f"   最终解范围: [{u_solution.min():.3f}, {u_solution.max():.3f}]")

    # 使用维度可视化器生成图像
    print(f"\\n🎨 生成伪时间步可视化结果...")

    visualizer = DimensionalVisualizer(config)
    data_test = {'x': x_coords}

    # 获取网格尺寸
    grid_size = int(np.sqrt(len(x_coords)))

    # 1. 主要解图 - 填色等高线
    print("1. 生成主解图...")
    save_path_main = 'pseudotime_main_solution.png'
    visualizer.plot_solution(u_solution, data_test, save_path_main,
                           plot_type='contourf',
                           colormap='RdBu_r',
                           levels=35,
                           grid_shape=(grid_size, grid_size),
                           figsize=(12, 9),
                           dpi=150)

    # 2. 高分辨率网格图
    print("2. 生成高分辨率网格图...")
    save_path_high = 'pseudotime_high_resolution.png'
    visualizer.plot_solution(u_solution, data_test, save_path_high,
                           plot_type='pcolormesh',
                           colormap='plasma',
                           grid_shape=(grid_size, grid_size),
                           figsize=(14, 10),
                           dpi=200)

    # 3. 等高线图
    print("3. 生成等高线图...")
    save_path_contour = 'pseudotime_contour.png'
    visualizer.plot_solution(u_solution, data_test, save_path_contour,
                           plot_type='contour',
                           colormap='viridis',
                           levels=25,
                           grid_shape=(grid_size, grid_size),
                           figsize=(12, 9),
                           dpi=150)

    # 4. 特殊视角 - 使用不同配色
    print("4. 生成特殊配色图...")
    save_path_special = 'pseudotime_special_view.png'
    visualizer.plot_solution(u_solution, data_test, save_path_special,
                           plot_type='contourf',
                           colormap='coolwarm',
                           levels=40,
                           grid_shape=(grid_size, grid_size),
                           figsize=(14, 10),
                           dpi=180)

    # 5. 伪时间步收敛历史图
    if 'residual_history' in locals():
        print("5. 生成收敛历史图...")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))
        iterations_range = list(range(1, len(residual_history) + 1))

        # 对数坐标
        ax.semilogy(iterations_range, residual_history, 'b-o', linewidth=3, markersize=8,
                   label='Solution Change', alpha=0.8)

        # 线性坐标 ( inset )
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="40%", height="30%", loc='upper right')
        axins.plot(iterations_range, residual_history, 'r-', linewidth=2)
        axins.set_xlabel('Iteration', fontsize=10)
        axins.set_ylabel('Change (linear)', fontsize=10)
        axins.grid(True, alpha=0.3)

        ax.set_xlabel('Pseudo-time Iteration', fontsize=14)
        ax.set_ylabel('Solution Change (log scale)', fontsize=14)
        ax.set_title('Pseudo-time Stepping Convergence', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        conv_save_path = os.path.join(config.results_dir, 'pseudotime_convergence.png')
        plt.savefig(conv_save_path, dpi=150, bbox_inches='tight')
        plt.close()

    # 6. 边界条件验证图
    print("6. 生成边界条件验证图...")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Boundary Conditions Validation', fontsize=16, fontweight='bold')

    x_flat = x_coords[:, 0]
    y_flat = x_coords[:, 1]

    # 左边界
    left_mask = (x_flat < -0.95)
    if np.any(left_mask):
        y_left = y_flat[left_mask]
        u_left = u_solution[left_mask]
        axes[0, 0].plot(y_left, u_left, 'b-', linewidth=2)
        axes[0, 0].axhline(y=-1.0, color='r', linestyle='--', linewidth=2, label='Target: -1.0')
        axes[0, 0].set_xlabel('y')
        axes[0, 0].set_ylabel('u')
        axes[0, 0].set_title('Left Boundary (x ≈ -1)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # 右边界
    right_mask = (x_flat > 0.95)
    if np.any(right_mask):
        y_right = y_flat[right_mask]
        u_right = u_solution[right_mask]
        axes[0, 1].plot(y_right, u_right, 'b-', linewidth=2)
        axes[0, 1].axhline(y=-1.0, color='r', linestyle='--', linewidth=2, label='Target: -1.0')
        axes[0, 1].set_xlabel('y')
        axes[0, 1].set_ylabel('u')
        axes[0, 1].set_title('Right Boundary (x ≈ 1)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 底部边界
    bottom_mask = (y_flat < 0.02)
    if np.any(bottom_mask):
        x_bottom = x_flat[bottom_mask]
        u_bottom = u_solution[bottom_mask]
        u_bottom_target = x_bottom**2 * np.cos(np.pi * x_bottom)
        axes[1, 0].plot(x_bottom, u_bottom, 'b-', linewidth=2, label='Computed')
        axes[1, 0].plot(x_bottom, u_bottom_target, 'r--', linewidth=2, label='Target: x²cos(πx)')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('u')
        axes[1, 0].set_title('Bottom Boundary (y ≈ 0)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 中心线剖面
    center_mask = (np.abs(y_flat - 0.5) < 0.02)
    if np.any(center_mask):
        x_center = x_flat[center_mask]
        u_center = u_solution[center_mask]
        axes[1, 1].plot(x_center, u_center, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('u')
        axes[1, 1].set_title('Solution Profile at y = 0.5')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    boundary_save_path = os.path.join(config.results_dir, 'pseudotime_boundary_validation.png')
    plt.savefig(boundary_save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 保存求解数据
    data_save_path = os.path.join(config.results_dir, 'pseudotime_visualization_data.npz')
    if 'residual_history' in locals():
        np.savez(data_save_path,
                 x_coords=x_coords,
                 u_solution=u_solution,
                 residual_history=residual_history,
                 X=X, Y=Y,
                 grid_size=grid_size)
        print(f"\\n💾 数据保存: {data_save_path}")

    # 总结结果
    print(f"\\n📊 伪时间步可视化结果总结:")

    result_files = [f for f in os.listdir(config.results_dir) if f.endswith('.png') and 'pseudotime' in f]
    total_size = 0
    for f in sorted(result_files):
        file_path = os.path.join(config.results_dir, f)
        size = os.path.getsize(file_path) / 1024
        total_size += size
        print(f"  📈 {f} - {size:.1f} KB")

    print(f"\\n📁 总大小: {total_size:.1f} KB")
    print(f"🎯 结果目录: {config.results_dir}")

    # 验证维度可视化器
    dim_info = visualizer.get_dimension_info()
    print(f"\\n✅ 维度可视化器验证:")
    print(f"   检测维度: {dim_info['n_dim']}D")
    print(f"   空间变量: {dim_info['spatial_vars']}")
    print(f"   推荐绘图: {dim_info['plot_type']}")

    print(f"\\n🎉 伪时间步算例可视化测试完成!")
    return True

if __name__ == "__main__":
    try:
        success = test_pseudotime_case()
        if success:
            print("\\n✅ 伪时间步算例可视化测试成功完成!")
        else:
            print("\\n❌ 可视化测试失败")
    except Exception as e:
        print(f"\\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()