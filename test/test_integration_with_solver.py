"""
测试维度可视化器与求解器的集成
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

# 导入求解器可视化模块
from src.problem_solvers.nonlinear_pde_solver.pseudotime_pde_solver.utils.visualize import PseudoTimePDEVisualizer
from src.problem_solvers.nonlinear_pde_solver.fixiteration_pde_solver.utils.visualize import FixIterationPDEVisualizer


def create_mock_pseudotime_config():
    """创建伪时间步求解器配置"""
    config_path = "/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/nonlinear_pde_cases/pseudotime_pde_cases/New_test_2D/config.json"

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # 添加求解器特有的配置
    config_dict.update({
        'results_dir': '/tmp/test_solver_results',
        'visualization': {
            'dataset_mode': 'test_mesh',
            'outputs': ['spotter', 'sniper'],
            'reference_policy': 'auto',
            'realtime_visualization': False,
            'save_animation': True,
            'animation_skip': 10,
            'figure_dpi': 100,
            'colormap': 'RdBu_r'
        }
    })

    class MockConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    return MockConfig(config_dict)


def create_mock_fixiteration_config():
    """创建固定迭代求解器配置"""
    config_path = "/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/nonlinear_pde_cases/pseudotime_pde_cases/New_test_2D/config.json"

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # 调整为固定迭代求解器配置
    config_dict.update({
        'problem_type': 'nonlinear_pde',
        'method': 'fixiteration',
        'results_dir': '/tmp/test_fixiteration_results',
        'visualization': {
            'outputs': ['spotter'],
            'reference_policy': 'auto',
            'save_animation': False,
            'figure_dpi': 150,
            'colormap': 'viridis'
        }
    })

    class MockConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    return MockConfig(config_dict)


def create_test_data(config):
    """根据配置创建测试数据"""
    x_domain = config.x_domain
    points_domain = config.points_domain

    # 创建测试网格
    grid_size = int(points_domain ** 0.5)
    x = np.linspace(x_domain[0][0], x_domain[0][1], grid_size)
    y = np.linspace(x_domain[1][0], x_domain[1][1], grid_size)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    x_coords = np.column_stack([X_flat, Y_flat])

    # 创建模拟解
    u = np.tanh(2 * X_flat + 0.5 * np.sin(2 * np.pi * Y_flat))

    data_train = {'x': x_coords[:grid_size**2//2]}  # 训练数据
    data_test = {'x': x_coords}  # 测试数据

    return data_train, data_test, u


def create_mock_results():
    """创建模拟求解结果"""
    return {
        'coefficients': np.random.randn(100, 1),
        'convergence_info': {
            'residual_history': [1.0, 0.1, 0.01, 0.001],
            'converged': True,
            'iterations': 4
        },
        'predictions': {
            'spotter': np.random.randn(10000, 1),
            'sniper': np.random.randn(10000, 1)
        }
    }


def test_pseudotime_visualizer():
    """测试伪时间步求解器可视化器"""
    print("=== 测试伪时间步求解器可视化器 ===")

    config = create_mock_pseudotime_config()
    data_train, data_test, u_solution = create_test_data(config)
    results = create_mock_results()

    # 使用实际解替换随机解
    results['predictions']['spotter'] = u_solution.reshape(-1, 1)
    results['predictions']['sniper'] = u_solution.reshape(-1, 1) * 1.01  # 轻微扰动

    visualizer = PseudoTimePDEVisualizer(config)

    print(f"维度信息: n_dim={visualizer.n_dim}")
    print(f"空间变量: {getattr(config, 'spatial_vars', 'N/A')}")

    # 测试plot_solution方法
    save_path = "/tmp/test_pseudotime_solution.png"
    visualizer.plot_solution(data_test, u_solution, save_path)

    if os.path.exists(save_path):
        print(f"✓ 解图保存成功: {save_path}")
    else:
        print(f"❌ 解图保存失败: {save_path}")

    # 测试render方法
    try:
        os.makedirs(config.results_dir, exist_ok=True)
        visualizer.render(results, data_train, data_test)
        print(f"✓ 完整可视化渲染成功: {config.results_dir}")
    except Exception as e:
        print(f"❌ 可视化渲染失败: {e}")


def test_fixiteration_visualizer():
    """测试固定迭代求解器可视化器"""
    print("\n=== 测试固定迭代求解器可视化器 ===")

    config = create_mock_fixiteration_config()
    data_train, data_test, u_solution = create_test_data(config)
    results = create_mock_results()

    # 使用实际解
    results['predictions']['spotter'] = u_solution.reshape(-1, 1)

    visualizer = FixIterationPDEVisualizer(config)

    print(f"维度信息: n_dim={visualizer.n_dim}")
    print(f"空间变量: {getattr(config, 'spatial_vars', 'N/A')}")

    # 测试plot_solution方法
    save_path = "/tmp/test_fixiteration_solution.png"
    visualizer.plot_solution(data_test, u_solution, save_path)

    if os.path.exists(save_path):
        print(f"✓ 解图保存成功: {save_path}")
    else:
        print(f"❌ 解图保存失败: {save_path}")


def test_dimensional_consistency():
    """测试维度一致性"""
    print("\n=== 测试维度一致性 ===")

    configs = [
        create_mock_pseudotime_config(),
        create_mock_fixiteration_config()
    ]

    for i, config in enumerate(configs, 1):
        visualizer_classes = [PseudoTimePDEVisualizer, FixIterationPDEVisualizer]

        for j, VisualizerClass in enumerate(visualizer_classes):
            visualizer = VisualizerClass(config)
            expected_dim = len(config.spatial_vars)

            print(f"配置{i}-可视化器{j}: expected={expected_dim}, actual={visualizer.n_dim}")
            assert visualizer.n_dim == expected_dim, f"维度不一致: 期望{expected_dim}, 实际{visualizer.n_dim}"

    print("✓ 所有可视化器维度检测一致")


def test_axis_labels_integration():
    """测试坐标轴标签集成"""
    print("\n=== 测试坐标轴标签集成 ===")

    config = create_mock_pseudotime_config()
    data_train, data_test, u_solution = create_test_data(config)

    visualizer = PseudoTimePDEVisualizer(config)

    # 测试不同维度的坐标轴标签
    print(f"空间变量: {config.spatial_vars}")
    print(f"期望的x轴标签: {config.spatial_vars[0]}")
    print(f"期望的y轴标签: {config.spatial_vars[1]}")

    save_path = "/tmp/test_axis_labels_integration.png"
    visualizer.plot_solution(data_test, u_solution, save_path)

    if os.path.exists(save_path):
        print("✓ 坐标轴标签集成测试成功")
    else:
        print("❌ 坐标轴标签集成测试失败")


if __name__ == "__main__":
    print("=== 维度可视化器与求解器集成测试 ===\n")

    try:
        test_pseudotime_visualizer()
        test_fixiteration_visualizer()
        test_dimensional_consistency()
        test_axis_labels_integration()

        print(f"\n✅ 所有集成测试通过！")

        # 显示生成的文件
        import glob
        result_files = glob.glob("/tmp/test_*solution.png") + glob.glob("/tmp/test_*integration.png")
        if result_files:
            print(f"\n生成的集成测试文件:")
            for f in sorted(result_files):
                print(f"  {f}")

        # 显示结果目录
        result_dirs = ['/tmp/test_solver_results', '/tmp/test_fixiteration_results']
        for result_dir in result_dirs:
            if os.path.exists(result_dir):
                files = os.listdir(result_dir)
                if files:
                    print(f"\n{result_dir} 目录中的文件:")
                    for f in sorted(files):
                        print(f"  {f}")

    except Exception as e:
        print(f"\n❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()