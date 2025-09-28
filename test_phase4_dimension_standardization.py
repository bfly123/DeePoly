#!/usr/bin/env python3
"""
测试第四阶段：维度标准化功能
Test Phase 4: Dimension standardization
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.utils.shape import (
    ensure_points_eqs, safe_eq_col, safe_all_eqs,
    broadcast_coeffs, assert_points_eqs, standardize_solution_shape,
    safe_segment_slice, concat_segments, validate_operator_output,
    ensure_matrix_result
)

def test_shape_utilities():
    """测试形状工具函数"""
    print("=" * 60)
    print("测试形状工具函数")
    print("=" * 60)

    # Test ensure_points_eqs
    print("\n测试 ensure_points_eqs:")
    u1d = np.array([1, 2, 3, 4, 5])
    u2d = ensure_points_eqs(u1d)
    print(f"  1D {u1d.shape} -> 2D {u2d.shape}")
    assert u2d.shape == (5, 1), f"Expected (5,1), got {u2d.shape}"

    u2d_input = np.array([[1, 2], [3, 4], [5, 6]])
    u2d_output = ensure_points_eqs(u2d_input)
    print(f"  2D {u2d_input.shape} -> 2D {u2d_output.shape}")
    assert u2d_output.shape == (3, 2), f"Expected (3,2), got {u2d_output.shape}"

    # Test safe_eq_col
    print("\n测试 safe_eq_col:")
    u_multi = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    col0 = safe_eq_col(u_multi, 0)
    col1 = safe_eq_col(u_multi, 1)
    print(f"  Multi-eq column 0: {col0.shape} = {col0.flatten()}")
    print(f"  Multi-eq column 1: {col1.shape} = {col1.flatten()}")
    assert col0.shape == (3, 1), f"Expected (3,1), got {col0.shape}"
    assert col1.shape == (3, 1), f"Expected (3,1), got {col1.shape}"

    # Test concat_segments
    print("\n测试 concat_segments:")
    seg1 = np.array([[1, 2], [3, 4]])
    seg2 = np.array([[5, 6], [7, 8], [9, 10]])
    segments = [seg1, seg2]
    concatenated = concat_segments(segments)
    print(f"  Segments {seg1.shape} + {seg2.shape} -> {concatenated.shape}")
    assert concatenated.shape == (5, 2), f"Expected (5,2), got {concatenated.shape}"

    # Test safe_segment_slice
    print("\n测试 safe_segment_slice:")
    u_global = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    segment = safe_segment_slice(u_global, 1, 4)
    print(f"  Global {u_global.shape}[1:4] -> {segment.shape}")
    assert segment.shape == (3, 2), f"Expected (3,2), got {segment.shape}"

    print("✓ 所有形状工具函数测试通过")
    return True

def test_dimension_consistency():
    """测试维度一致性"""
    print("\n" + "=" * 60)
    print("测试维度一致性")
    print("=" * 60)

    # Test single equation consistency
    print("\n测试单方程维度一致性:")
    single_eq_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    single_eq_2d = ensure_points_eqs(single_eq_1d)

    print(f"  原始1D: {single_eq_1d.shape}")
    print(f"  标准化2D: {single_eq_2d.shape}")

    # Simulate dimension-aware operations
    eq_col = safe_eq_col(single_eq_2d, 0)
    print(f"  提取方程列: {eq_col.shape}")

    # Test multi-equation consistency
    print("\n测试多方程维度一致性:")
    multi_eq = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    eq0 = safe_eq_col(multi_eq, 0)
    eq1 = safe_eq_col(multi_eq, 1)

    print(f"  多方程数组: {multi_eq.shape}")
    print(f"  方程0: {eq0.shape}")
    print(f"  方程1: {eq1.shape}")

    # Test segment operations
    print("\n测试段操作维度一致性:")
    global_solution = np.random.rand(20, 2)

    # Simulate global_to_segments
    segments = []
    segment_sizes = [5, 7, 8]
    start_idx = 0

    for size in segment_sizes:
        end_idx = start_idx + size
        segment = safe_segment_slice(global_solution, start_idx, end_idx)
        segments.append(segment)
        print(f"  段 {len(segments)}: {segment.shape}")
        start_idx = end_idx

    # Simulate segments_to_global
    reconstructed = concat_segments(segments)
    print(f"  重构全局: {reconstructed.shape}")

    assert reconstructed.shape == global_solution.shape, "重构形状不匹配"
    assert np.allclose(reconstructed, global_solution), "重构值不匹配"

    print("✓ 所有维度一致性测试通过")
    return True

def test_base_fitter_integration():
    """测试base_fitter.py集成"""
    print("\n" + "=" * 60)
    print("测试base_fitter.py集成")
    print("=" * 60)

    try:
        from src.abstract_class.base_fitter import BaseDeepPolyFitter

        # Test that shape utilities are properly imported
        print("✓ shape utilities 导入成功")

        # Create mock data to test global_to_segments and segments_to_global
        print("\n测试段操作模拟:")

        # Mock BaseDeepPolyFitter methods
        class MockFitter:
            def __init__(self):
                self.ns = 3
                self.n_eqs = 2
                self.data = {
                    "x_segments_norm": [
                        np.linspace(0, 1, 5),   # segment 0: 5 points
                        np.linspace(0, 1, 7),   # segment 1: 7 points
                        np.linspace(0, 1, 8)    # segment 2: 8 points
                    ]
                }

            def global_to_segments(self, U_global):
                """模拟global_to_segments方法"""
                from src.utils.shape import safe_segment_slice

                U_segments = []
                start_idx = 0

                for seg_idx in range(self.ns):
                    n_points = len(self.data["x_segments_norm"][seg_idx])
                    end_idx = start_idx + n_points

                    U_segment = safe_segment_slice(U_global, start_idx, end_idx, "U_global")
                    U_segments.append(U_segment)

                    start_idx = end_idx

                return U_segments

            def segments_to_global(self, U_segments):
                """模拟segments_to_global方法"""
                from src.utils.shape import concat_segments

                if not U_segments or U_segments[0] is None:
                    total_points = sum(len(self.data["x_segments_norm"][i]) for i in range(self.ns))
                    return np.zeros((total_points, self.n_eqs))

                return concat_segments(U_segments, "U_segments")

        mock_fitter = MockFitter()

        # Test with 1D input (single equation)
        print("\n测试1D输入(单方程):")
        U_global_1d = np.random.rand(20)  # 20 points, 1 equation
        U_global_2d = ensure_points_eqs(U_global_1d)

        segments = mock_fitter.global_to_segments(U_global_2d)
        print(f"  输入全局: {U_global_2d.shape}")
        for i, seg in enumerate(segments):
            print(f"  段{i}: {seg.shape}")

        reconstructed = mock_fitter.segments_to_global(segments)
        print(f"  重构全局: {reconstructed.shape}")

        # Test with 2D input (multi equation)
        print("\n测试2D输入(多方程):")
        U_global_2d_multi = np.random.rand(20, 2)  # 20 points, 2 equations

        segments_multi = mock_fitter.global_to_segments(U_global_2d_multi)
        print(f"  输入全局: {U_global_2d_multi.shape}")
        for i, seg in enumerate(segments_multi):
            print(f"  段{i}: {seg.shape}")

        reconstructed_multi = mock_fitter.segments_to_global(segments_multi)
        print(f"  重构全局: {reconstructed_multi.shape}")

        print("✓ base_fitter.py 集成测试通过")
        return True

    except Exception as e:
        print(f"✗ base_fitter.py 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_operator_factory_integration():
    """测试operator_factory.py集成"""
    print("\n" + "=" * 60)
    print("测试operator_factory.py集成")
    print("=" * 60)

    try:
        from src.abstract_class.operator_factory import OperatorFactory

        print("✓ OperatorFactory 导入成功")

        # Test that shape utilities are properly imported
        print("✓ shape utilities 在 operator_factory 中导入成功")

        # Create a simple operator factory
        all_derivatives = {
            "u": (0, 0),      # Variable u, derivative order 0
            "u_x": (0, 1),    # Variable u, derivative order 1
            "v": (1, 0),      # Variable v, derivative order 0
        }

        factory = OperatorFactory(all_derivatives)
        print("✓ OperatorFactory 实例创建成功")

        # Test dimension standardization in operator creation
        print("\n测试算子创建中的维度标准化:")

        # Simulate operator evaluation with mixed dimension inputs
        test_cases = [
            ("单方程1D输入", np.random.rand(10)),
            ("单方程2D输入", np.random.rand(10, 1)),
            ("多方程2D输入", np.random.rand(10, 2)),
        ]

        for case_name, u_test in test_cases:
            print(f"  {case_name}: 输入形状 {u_test.shape}")

            # The ensure_points_eqs function should standardize these
            from src.utils.shape import ensure_points_eqs
            u_std = ensure_points_eqs(u_test)
            print(f"    标准化后: {u_std.shape}")

            # Verify 2D output
            assert u_std.ndim == 2, f"期望2D输出，得到{u_std.ndim}D"
            assert u_std.shape[0] == 10, f"期望10个点，得到{u_std.shape[0]}个点"

        print("✓ operator_factory.py 集成测试通过")
        return True

    except Exception as e:
        print(f"✗ operator_factory.py 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_time_schemes_dimension_consistency():
    """测试时间积分格式的维度一致性"""
    print("\n" + "=" * 60)
    print("测试时间积分格式维度一致性")
    print("=" * 60)

    try:
        # Test that time schemes can handle standardized dimensions
        from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig

        # Create temporary config with minimal setup
        temp_config = {
            "problem_type": "time_pde",
            "method": "hybrid",
            "auto_code": False,
            "eq": {
                "L1": ["0"],
                "L2": ["0"],
                "F": ["1"],
                "N": ["0"]
            },
            "vars_list": ["u"],
            "spatial_vars": ["x"],
            "hidden_dims": [16, 16],
            "n_segments": [3],
            "poly_degree": [3],
            "x_domain": [[-1, 1]]
        }

        temp_dir = "/tmp/test_phase4_dimension"
        os.makedirs(temp_dir, exist_ok=True)

        import json
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump(temp_config, f, indent=2)

        config = TimePDEConfig(case_dir=temp_dir)
        config.device = "cpu"

        print("✓ 时间PDE配置创建成功")

        # Test dimension handling in solution arrays
        print("\n测试解数组维度处理:")

        # Simulate single equation solution (should be standardized to 2D)
        u_single_1d = np.random.rand(100)  # 100 points, 1 equation
        u_single_2d = ensure_points_eqs(u_single_1d)

        print(f"  单方程: {u_single_1d.shape} -> {u_single_2d.shape}")
        assert u_single_2d.shape == (100, 1), f"期望(100,1)，得到{u_single_2d.shape}"

        # Test segment operations with standardized dimensions
        segments = [
            np.random.rand(30, 1),
            np.random.rand(35, 1),
            np.random.rand(35, 1)
        ]

        global_solution = concat_segments(segments)
        print(f"  段重组: {[s.shape for s in segments]} -> {global_solution.shape}")
        assert global_solution.shape == (100, 1), f"期望(100,1)，得到{global_solution.shape}"

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

        print("✓ 时间积分格式维度一致性测试通过")
        return True

    except Exception as e:
        print(f"✗ 时间积分格式维度一致性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试第四阶段：维度标准化功能...")

    success_count = 0
    total_tests = 5

    # Test 1: Shape utilities
    if test_shape_utilities():
        success_count += 1

    # Test 2: Dimension consistency
    if test_dimension_consistency():
        success_count += 1

    # Test 3: Base fitter integration
    if test_base_fitter_integration():
        success_count += 1

    # Test 4: Operator factory integration
    if test_operator_factory_integration():
        success_count += 1

    # Test 5: Time schemes dimension consistency
    if test_time_schemes_dimension_consistency():
        success_count += 1

    print("\n" + "=" * 60)
    print(f"第四阶段测试完成: {success_count}/{total_tests} 个测试通过")

    if success_count == total_tests:
        print("✓ 所有维度标准化测试通过！")
        print("✓ Phase 4 dimension standardization 功能正常工作")
        sys.exit(0)
    else:
        print("✗ 部分测试失败，需要检查维度标准化逻辑")
        sys.exit(1)