"""
测试New_test_2D算例的维度可视化
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from src.visualization.dimensional_plotter import DimensionalVisualizer


def load_config_and_create_test_data():
    """加载配置并创建测试数据"""
    config_path = "/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/cases/nonlinear_pde_cases/pseudotime_pde_cases/New_test_2D/config.json"

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # 创建模拟配置对象
    class MockConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    config = MockConfig(config_dict)

    # 根据配置创建测试数据
    x_domain = config.x_domain
    n_segments = config.n_segments
    points_domain = config.points_domain

    print(f"配置信息:")
    print(f"  spatial_vars: {config.spatial_vars}")
    print(f"  x_domain: {x_domain}")
    print(f"  n_segments: {n_segments}")
    print(f"  points_domain: {points_domain}")

    # 创建测试网格
    x = np.linspace(x_domain[0][0], x_domain[0][1], int(points_domain ** 0.5))
    y = np.linspace(x_domain[1][0], x_domain[1][1], int(points_domain ** 0.5))
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    x_coords = np.column_stack([X_flat, Y_flat])

    # 创建模拟解场 (基于Allen-Cahn方程的解)
    u = np.tanh(2 * X_flat) * np.cos(np.pi * Y_flat)

    data = {'x': x_coords}

    return config, data, u


def test_dimension_detection():
    """测试维度检测"""
    config, data, u = load_config_and_create_test_data()

    visualizer = DimensionalVisualizer(config)
    dim_info = visualizer.get_dimension_info()

    print(f"\n维度检测结果:")
    print(f"  n_dim: {dim_info['n_dim']}")
    print(f"  spatial_vars: {dim_info['spatial_vars']}")
    print(f"  plot_type: {dim_info['plot_type']}")

    assert dim_info['n_dim'] == 2, f"期望2维，实际{dim_info['n_dim']}维"
    assert dim_info['spatial_vars'] == ['x', 'y'], f"期望['x', 'y']，实际{dim_info['spatial_vars']}"
    print("✓ 维度检测正确")


def test_2d_solution_plotting():
    """测试2D解绘图"""
    config, data, u = load_config_and_create_test_data()
    visualizer = DimensionalVisualizer(config)

    # 测试不同绘图类型
    plot_types = ['contourf', 'contour', 'pcolormesh']

    for plot_type in plot_types:
        save_path = f"/tmp/test_new_test_2d_{plot_type}.png"

        print(f"\n测试{plot_type}绘图...")

        viz_config = {
            'plot_type': plot_type,
            'figsize': (12, 8),
            'colormap': 'RdBu_r',
            'levels': 30,
            'grid_shape': (int(len(data['x']) ** 0.5), int(len(data['x']) ** 0.5))
        }

        try:
            visualizer.plot_solution(u, data, save_path, **viz_config)
            assert os.path.exists(save_path), f"绘图文件未生成: {save_path}"
            print(f"✓ {plot_type}绘图成功: {save_path}")
        except Exception as e:
            print(f"❌ {plot_type}绘图失败: {e}")
            raise


def test_custom_visualization():
    """测试自定义可视化配置"""
    config, data, u = load_config_and_create_test_data()
    visualizer = DimensionalVisualizer(config)

    # 使用配置中的可视化参数
    if hasattr(config, 'visualization'):
        viz_config = {
            'colormap': config.visualization.get('colormap', 'RdBu_r'),
            'figure_dpi': config.visualization.get('figure_dpi', 100),
            'grid_shape': (100, 100)
        }

        save_path = "/tmp/test_new_test_2d_custom.png"
        visualizer.plot_solution(u, data, save_path, **viz_config)

        assert os.path.exists(save_path), "自定义配置绘图失败"
        print(f"✓ 自定义配置绘图成功: {save_path}")


def test_axis_labels():
    """测试坐标轴标签"""
    config, data, u = load_config_and_create_test_data()
    visualizer = DimensionalVisualizer(config)

    print(f"\n坐标轴标签测试:")
    print(f"  x轴标签应为: {config.spatial_vars[0]}")
    print(f"  y轴标签应为: {config.spatial_vars[1]}")

    save_path = "/tmp/test_new_test_2d_labels.png"
    visualizer.plot_solution(u, data, save_path, grid_shape=(int(len(data['x']) ** 0.5), int(len(data['x']) ** 0.5)))

    assert os.path.exists(save_path), "坐标轴标签测试失败"
    print("✓ 坐标轴标签正确应用")


def create_sample_solution():
    """创建更真实的解场"""
    config, data, _ = load_config_and_create_test_data()

    # 创建Allen-Cahn方程的稳态解近似
    x = data['x'][:, 0]
    y = data['x'][:, 1]

    # 界面解 + 扰动
    u = np.tanh(2 * x + 0.5 * np.sin(2 * np.pi * y))

    return u


def test_realistic_solution():
    """测试更真实的解场可视化"""
    config, data, _ = load_config_and_create_test_data()
    u = create_sample_solution()
    visualizer = DimensionalVisualizer(config)

    save_path = "/tmp/test_new_test_2d_realistic.png"
    viz_config = {
        'plot_type': 'contourf',
        'colormap': 'seismic',
        'levels': 40,
        'grid_shape': (int(len(data['x']) ** 0.5), int(len(data['x']) ** 0.5)),
        'figsize': (14, 10)
    }

    visualizer.plot_solution(u, data, save_path, **viz_config)

    assert os.path.exists(save_path), "真实解场可视化失败"
    print(f"✓ 真实解场可视化成功: {save_path}")


if __name__ == "__main__":
    print("=== New_test_2D 算例维度可视化测试 ===\n")

    try:
        # 加载配置并显示信息
        config, data, u = load_config_and_create_test_data()

        # 运行测试
        test_dimension_detection()
        test_2d_solution_plotting()
        test_custom_visualization()
        test_axis_labels()
        test_realistic_solution()

        print(f"\n✅ 所有测试通过！")
        print(f"生成的测试图片位于 /tmp/ 目录")

        # 列出生成的文件
        import glob
        test_files = glob.glob("/tmp/test_new_test_2d_*.png")
        if test_files:
            print(f"\n生成的测试文件:")
            for f in sorted(test_files):
                print(f"  {f}")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()