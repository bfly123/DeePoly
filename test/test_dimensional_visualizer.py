"""
测试维度可视化器
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.dimensional_plotter import DimensionalVisualizer


class MockConfig1D:
    def __init__(self):
        self.spatial_vars = ["x"]


class MockConfig2D:
    def __init__(self):
        self.spatial_vars = ["x", "y"]


def test_1d_visualization():
    """测试1D可视化"""
    print("Testing 1D visualization...")

    config = MockConfig1D()
    visualizer = DimensionalVisualizer(config)

    # 创建1D测试数据
    x = np.linspace(-1, 1, 100).reshape(-1, 1)
    u = np.sin(np.pi * x)
    data = {'x': x}

    # 测试绘图
    save_path = "/tmp/test_1d_solution.png"
    visualizer.plot_solution(u, data, save_path)

    # 检查维度信息
    dim_info = visualizer.get_dimension_info()
    print(f"1D Dimension info: {dim_info}")

    assert dim_info['n_dim'] == 1
    assert os.path.exists(save_path)
    print("✓ 1D visualization test passed")


def test_2d_visualization():
    """测试2D可视化"""
    print("Testing 2D visualization...")

    config = MockConfig2D()
    visualizer = DimensionalVisualizer(config)

    # 创建2D测试数据
    x = np.linspace(-1, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    x_coords = np.column_stack([X_flat, Y_flat])

    # 创建测试解场
    u = np.sin(np.pi * X_flat) * np.cos(2 * np.pi * Y_flat)
    data = {'x': x_coords}

    # 测试绘图
    save_path = "/tmp/test_2d_solution.png"
    visualizer.plot_solution(u, data, save_path, grid_shape=(50, 50))

    # 检查维度信息
    dim_info = visualizer.get_dimension_info()
    print(f"2D Dimension info: {dim_info}")

    assert dim_info['n_dim'] == 2
    assert os.path.exists(save_path)
    print("✓ 2D visualization test passed")


def test_reference_comparison():
    """测试参考解对比"""
    print("Testing reference comparison...")

    config = MockConfig1D()
    visualizer = DimensionalVisualizer(config)

    # 创建数值解和参考解
    x_num = np.linspace(-1, 1, 100).reshape(-1, 1)
    u_num = np.sin(np.pi * x_num) + 0.01 * np.random.randn(100, 1)

    x_ref = np.linspace(-1, 1, 200).reshape(-1, 1)
    u_ref = np.sin(np.pi * x_ref)

    data = {'x': x_num}
    reference_solution = {'x_ref': x_ref, 'u_ref': u_ref}

    save_path = "/tmp/test_comparison.png"
    visualizer.plot_comparison_with_reference(u_num, data, reference_solution, save_path)

    assert os.path.exists(save_path)
    print("✓ Reference comparison test passed")


if __name__ == "__main__":
    try:
        test_1d_visualization()
        test_2d_visualization()
        test_reference_comparison()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()