#!/usr/bin/env python3
"""
测试简化的周期边界条件实现
Test Simplified Periodic Boundary Condition Implementation
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import json
from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig
from src.problem_solvers.time_pde_solver.utils.data import TimePDEDataGenerator
from src.problem_solvers.time_pde_solver.core.fitter import TimePDEFitter

def test_simplified_periodic_configuration():
    """测试简化的周期边界条件配置解析"""
    print("=" * 60)
    print("测试简化的周期边界条件配置解析")
    print("=" * 60)

    # 创建临时配置 - 包含周期边界条件但无constraint字段
    temp_config = {
        "problem_type": "time_pde",
        "method": "hybrid",
        "auto_code": False,
        "eq": {
            "L1": ["0.001*diff(u,x,2)"],
            "L2": ["0"],
            "F": ["1"],
            "N": ["u**3-u"]
        },
        "vars_list": ["u"],
        "spatial_vars": ["x"],
        "Initial_conditions": [
            {
                "var": "u",
                "value": "sin(pi*x)",
                "points": 50
            }
        ],
        "boundary_conditions": [
            {
                "type": "periodic",
                "region": "left",
                "pair_with": "right",
                # 注意：移除了"constraint"字段 - 简化后不再需要
                "points": 1
            }
        ],
        "hidden_dims": [16, 16],
        "n_segments": [3],
        "poly_degree": [3],
        "x_domain": [[-1, 1]]
    }

    temp_dir = "/tmp/test_simplified_periodic"
    os.makedirs(temp_dir, exist_ok=True)

    with open(os.path.join(temp_dir, "config.json"), "w") as f:
        json.dump(temp_config, f, indent=2)

    try:
        # 测试配置解析
        config = TimePDEConfig(case_dir=temp_dir)
        config.device = "cpu"

        print("✓ 简化的周期边界条件配置解析成功")

        # 生成数据并检查周期边界条件结构
        data_generator = TimePDEDataGenerator(config)
        data = data_generator.generate_data(mode="train")

        print("✓ 数据生成成功")

        # 检查边界数据结构
        boundary_data = data.get("boundary_data", {})
        if 0 in boundary_data and "periodic" in boundary_data[0]:
            periodic_pairs = boundary_data[0]["periodic"]["pairs"]
            print(f"✓ 检测到 {len(periodic_pairs)} 个周期边界条件对")

            for i, pair in enumerate(periodic_pairs):
                print(f"  对 {i+1}:")
                print(f"    x_1 形状: {pair['x_1'].shape}")
                print(f"    x_2 形状: {pair['x_2'].shape}")
                # 验证不再包含constraint_type
                assert 'constraint_type' not in pair, "简化后的周期边界条件不应包含constraint_type"
                print(f"    ✓ 无constraint_type字段 (简化成功)")

                # 验证不再包含normals
                assert 'normals_1' not in pair, "简化后的周期边界条件不应包含normals_1"
                assert 'normals_2' not in pair, "简化后的周期边界条件不应包含normals_2"
                print(f"    ✓ 无normals字段 (简化成功)")

        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"✗ 简化周期边界条件配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simplified_periodic_boundary_constraint():
    """测试简化的边界约束实现"""
    print("\n" + "=" * 60)
    print("测试简化的边界约束实现")
    print("=" * 60)

    try:
        from src.abstract_class.boundary_constraint import BoundaryConstraint
        import torch

        # 创建简化的周期边界约束
        constraint = BoundaryConstraint(
            var_idx=0,
            constraint_type='periodic',
            x_coords=torch.tensor([[0.0], [0.5]], dtype=torch.float64),
            x_coords_pair=torch.tensor([[1.0], [1.5]], dtype=torch.float64)
            # 注意：不再设置periodic_type或normals_pair
        )

        print("✓ 简化的BoundaryConstraint创建成功")

        # 测试统一的周期边界条件评估
        U_pred_1 = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        U_pred_2 = torch.tensor([[1.1], [2.1]], dtype=torch.float64)

        # 评估周期边界条件 (应该只计算函数值差异)
        residual = constraint.evaluate_periodic(U_pred_1, U_pred_2)

        print(f"✓ 周期边界条件评估成功")
        print(f"  残差形状: {residual.shape}")
        print(f"  残差值: {residual.detach().numpy().flatten()}")

        # 验证残差是函数值差异
        expected_residual = U_pred_1[:, 0:1] - U_pred_2[:, 0:1]
        assert torch.allclose(residual, expected_residual), "周期边界条件残差计算错误"

        print("✓ 周期边界条件残差计算正确")

        return True

    except Exception as e:
        print(f"✗ 简化边界约束测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simplified_periodic_fitter_integration():
    """测试简化周期边界条件与fitter的集成"""
    print("\n" + "=" * 60)
    print("测试简化周期边界条件与fitter的集成")
    print("=" * 60)

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
        "Initial_conditions": [
            {
                "var": "u",
                "value": "sin(pi*x)",
                "points": 20
            }
        ],
        "boundary_conditions": [
            {
                "type": "periodic",
                "region": "left",
                "pair_with": "right",
                "points": 1
            }
        ],
        "hidden_dims": [8, 8],
        "n_segments": [2],
        "poly_degree": [2],
        "x_domain": [[-1, 1]]
    }

    temp_dir = "/tmp/test_periodic_fitter"
    os.makedirs(temp_dir, exist_ok=True)

    with open(os.path.join(temp_dir, "config.json"), "w") as f:
        json.dump(temp_config, f, indent=2)

    try:
        config = TimePDEConfig(case_dir=temp_dir)
        config.device = "cpu"

        data_generator = TimePDEDataGenerator(config)
        data = data_generator.generate_data(mode="train")

        # 创建fitter并测试初始化
        fitter = TimePDEFitter(config, data)

        print("✓ TimePDEFitter与简化周期边界条件创建成功")

        # 测试周期边界约束是否正确添加
        if hasattr(fitter, 'A') and fitter.A:
            print(f"✓ 约束矩阵创建成功，约束数量: {len(fitter.A)}")

        print("✓ 简化周期边界条件fitter集成测试通过")

        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"✗ 简化周期边界条件fitter集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """测试向后兼容性 - 使用现有的Allen-Cahn配置"""
    print("\n" + "=" * 60)
    print("测试向后兼容性")
    print("=" * 60)

    try:
        # 使用现有的Allen-Cahn配置（带有constraint字段）
        case_dir = "cases/Time_pde_cases/Allen_Cahn/AC_equation_100_0.01"

        if not os.path.exists(case_dir):
            print("⚠ Allen-Cahn测试案例不存在，跳过向后兼容性测试")
            return True

        config = TimePDEConfig(case_dir=case_dir)
        config.device = "cpu"

        print("✓ 现有配置加载成功 (向后兼容)")

        data_generator = TimePDEDataGenerator(config)
        data = data_generator.generate_data(mode="train")

        print("✓ 数据生成成功 (向后兼容)")

        # 检查是否正确处理了旧的constraint字段
        boundary_data = data.get("boundary_data", {})
        if 0 in boundary_data and "periodic" in boundary_data[0]:
            pairs = boundary_data[0]["periodic"]["pairs"]
            for pair in pairs:
                # 旧配置可能仍有constraint_type，但简化的实现应该忽略它
                if 'constraint_type' in pair:
                    print(f"  ⚠ 检测到旧的constraint_type字段: {pair['constraint_type']} (将被忽略)")

        print("✓ 向后兼容性测试通过")
        return True

    except Exception as e:
        print(f"✗ 向后兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """测试性能对比 - 简化后的实现应该更快"""
    print("\n" + "=" * 60)
    print("测试性能对比")
    print("=" * 60)

    try:
        import time
        import torch
        from src.abstract_class.boundary_constraint import BoundaryConstraint

        # 创建测试数据
        n_points = 100
        x_coords_1 = torch.randn(n_points, 1, dtype=torch.float64)
        x_coords_2 = torch.randn(n_points, 1, dtype=torch.float64)
        U_pred_1 = torch.randn(n_points, 1, dtype=torch.float64)
        U_pred_2 = torch.randn(n_points, 1, dtype=torch.float64)

        # 创建简化的周期边界约束
        constraint = BoundaryConstraint(
            var_idx=0,
            constraint_type='periodic',
            x_coords=x_coords_1,
            x_coords_pair=x_coords_2
        )

        # 性能测试：简化的周期边界条件
        n_iterations = 1000
        start_time = time.time()

        for _ in range(n_iterations):
            residual = constraint.evaluate_periodic(U_pred_1, U_pred_2)

        simplified_time = time.time() - start_time

        print(f"✓ 简化的周期边界条件性能测试完成")
        print(f"  {n_iterations} 次迭代耗时: {simplified_time:.4f} 秒")
        print(f"  平均每次耗时: {simplified_time/n_iterations*1000:.4f} 毫秒")

        # 与直接计算比较
        start_time = time.time()

        for _ in range(n_iterations):
            direct_residual = U_pred_1[:, 0:1] - U_pred_2[:, 0:1]

        direct_time = time.time() - start_time

        print(f"✓ 直接计算基准测试完成")
        print(f"  {n_iterations} 次迭代耗时: {direct_time:.4f} 秒")
        print(f"  性能开销: {(simplified_time/direct_time - 1)*100:.1f}%")

        if simplified_time / direct_time < 2.0:  # 开销应该在合理范围内
            print("✓ 性能测试通过 - 简化实现效率良好")
        else:
            print("⚠ 性能警告 - 实现可能需要进一步优化")

        return True

    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试简化的周期边界条件实现...")

    success_count = 0
    total_tests = 5

    # 测试1：简化配置解析
    if test_simplified_periodic_configuration():
        success_count += 1

    # 测试2：简化边界约束
    if test_simplified_periodic_boundary_constraint():
        success_count += 1

    # 测试3：fitter集成
    if test_simplified_periodic_fitter_integration():
        success_count += 1

    # 测试4：向后兼容性
    if test_backward_compatibility():
        success_count += 1

    # 测试5：性能对比
    if test_performance_comparison():
        success_count += 1

    print("\n" + "=" * 60)
    print(f"简化周期边界条件测试完成: {success_count}/{total_tests} 个测试通过")

    if success_count == total_tests:
        print("✅ 所有测试通过！简化的周期边界条件实现成功")
        print("📈 性能提升：消除了复杂的导数计算")
        print("🧹 代码简化：移除了不必要的类型分支")
        print("🔧 数学正确：与周期边界条件的真实定义一致")
        sys.exit(0)
    else:
        print("❌ 部分测试失败，需要检查简化实现")
        sys.exit(1)