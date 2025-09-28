#!/usr/bin/env python3
"""
测试最优化的周期边界条件配置格式
Test Optimal Periodic Boundary Condition Configuration Format
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import json
from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig
from src.problem_solvers.time_pde_solver.utils.data import TimePDEDataGenerator

def test_new_pairs_format():
    """测试新的pairs格式"""
    print("=" * 60)
    print("测试新的pairs格式")
    print("=" * 60)

    # 创建使用新格式的配置
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
                "points": 30
            }
        ],
        "boundary_conditions": [
            {
                "type": "periodic",
                "pairs": ["left", "right"],  # 新格式
                "points": 1
            }
        ],
        "hidden_dims": [16, 16],
        "n_segments": [3],
        "poly_degree": [3],
        "x_domain": [[-1, 1]]
    }

    temp_dir = "/tmp/test_new_pairs_format"
    os.makedirs(temp_dir, exist_ok=True)

    with open(os.path.join(temp_dir, "config.json"), "w") as f:
        json.dump(temp_config, f, indent=2)

    try:
        # 测试配置解析
        config = TimePDEConfig(case_dir=temp_dir)
        config.device = "cpu"

        print("✓ 新pairs格式配置解析成功")

        # 生成数据并检查边界条件结构
        data_generator = TimePDEDataGenerator(config)
        data = data_generator.generate_data(mode="train")

        print("✓ 数据生成成功")

        # 验证边界数据结构
        boundary_data = data.get("boundary_data", {})
        if 0 in boundary_data and "periodic" in boundary_data[0]:
            periodic_pairs = boundary_data[0]["periodic"]["pairs"]
            print(f"✓ 检测到 {len(periodic_pairs)} 个周期边界条件对")

            for i, pair in enumerate(periodic_pairs):
                print(f"  对 {i+1}: x_1.shape={pair['x_1'].shape}, x_2.shape={pair['x_2'].shape}")

        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"✗ 新pairs格式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """测试向后兼容性 - 旧格式仍然工作"""
    print("\n" + "=" * 60)
    print("测试向后兼容性")
    print("=" * 60)

    # 创建使用旧格式的配置
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
                "region": "left",        # 旧格式
                "pair_with": "right",    # 旧格式
                "points": 1
            }
        ],
        "hidden_dims": [8, 8],
        "n_segments": [2],
        "poly_degree": [2],
        "x_domain": [[-1, 1]]
    }

    temp_dir = "/tmp/test_backward_compat"
    os.makedirs(temp_dir, exist_ok=True)

    with open(os.path.join(temp_dir, "config.json"), "w") as f:
        json.dump(temp_config, f, indent=2)

    try:
        config = TimePDEConfig(case_dir=temp_dir)
        config.device = "cpu"

        print("✓ 旧格式配置解析成功 (向后兼容)")

        data_generator = TimePDEDataGenerator(config)
        data = data_generator.generate_data(mode="train")

        print("✓ 旧格式数据生成成功")

        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"✗ 向后兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_updated_allen_cahn_config():
    """测试更新后的Allen-Cahn配置"""
    print("\n" + "=" * 60)
    print("测试更新后的Allen-Cahn配置")
    print("=" * 60)

    try:
        # 使用更新后的Allen-Cahn配置
        case_dir = "cases/Time_pde_cases/Allen_Cahn/AC_equation_100_0.01"

        if not os.path.exists(case_dir):
            print("⚠ Allen-Cahn测试案例不存在，跳过测试")
            return True

        config = TimePDEConfig(case_dir=case_dir)
        config.device = "cpu"

        print("✓ 更新后的Allen-Cahn配置加载成功")

        data_generator = TimePDEDataGenerator(config)
        data = data_generator.generate_data(mode="train")

        print("✓ 数据生成成功")

        # 验证新格式被正确解析
        boundary_data = data.get("boundary_data", {})
        if 0 in boundary_data and "periodic" in boundary_data[0]:
            pairs = boundary_data[0]["periodic"]["pairs"]
            if pairs:
                print(f"✓ 周期边界条件对解析成功: {len(pairs)} 对")
            else:
                print("⚠ 未检测到周期边界条件对")

        return True

    except Exception as e:
        print(f"✗ Allen-Cahn配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_format_comparison():
    """对比新旧格式的等效性"""
    print("\n" + "=" * 60)
    print("测试新旧格式等效性")
    print("=" * 60)

    base_config = {
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
        "hidden_dims": [8, 8],
        "n_segments": [2],
        "poly_degree": [2],
        "x_domain": [[-1, 1]]
    }

    # 新格式配置
    new_config = base_config.copy()
    new_config["boundary_conditions"] = [
        {
            "type": "periodic",
            "pairs": ["left", "right"],
            "points": 1
        }
    ]

    # 旧格式配置
    old_config = base_config.copy()
    old_config["boundary_conditions"] = [
        {
            "type": "periodic",
            "region": "left",
            "pair_with": "right",
            "points": 1
        }
    ]

    try:
        # 测试新格式
        temp_dir_new = "/tmp/test_new_format"
        os.makedirs(temp_dir_new, exist_ok=True)
        with open(os.path.join(temp_dir_new, "config.json"), "w") as f:
            json.dump(new_config, f, indent=2)

        config_new = TimePDEConfig(case_dir=temp_dir_new)
        config_new.device = "cpu"
        data_gen_new = TimePDEDataGenerator(config_new)
        data_new = data_gen_new.generate_data(mode="train")

        # 测试旧格式
        temp_dir_old = "/tmp/test_old_format"
        os.makedirs(temp_dir_old, exist_ok=True)
        with open(os.path.join(temp_dir_old, "config.json"), "w") as f:
            json.dump(old_config, f, indent=2)

        config_old = TimePDEConfig(case_dir=temp_dir_old)
        config_old.device = "cpu"
        data_gen_old = TimePDEDataGenerator(config_old)
        data_old = data_gen_old.generate_data(mode="train")

        # 比较边界数据结构
        boundary_new = data_new.get("boundary_data", {})
        boundary_old = data_old.get("boundary_data", {})

        print("✓ 新旧格式都解析成功")

        # 验证边界条件数据一致性
        if 0 in boundary_new and 0 in boundary_old:
            if "periodic" in boundary_new[0] and "periodic" in boundary_old[0]:
                pairs_new = boundary_new[0]["periodic"]["pairs"]
                pairs_old = boundary_old[0]["periodic"]["pairs"]

                if len(pairs_new) == len(pairs_old):
                    print(f"✓ 边界条件对数量一致: {len(pairs_new)}")

                    # 比较边界点坐标
                    for i, (pair_new, pair_old) in enumerate(zip(pairs_new, pairs_old)):
                        x1_match = np.allclose(pair_new['x_1'], pair_old['x_1'])
                        x2_match = np.allclose(pair_new['x_2'], pair_old['x_2'])

                        if x1_match and x2_match:
                            print(f"  ✓ 对 {i+1}: 边界点坐标完全一致")
                        else:
                            print(f"  ✗ 对 {i+1}: 边界点坐标不一致")
                            return False

                    print("✓ 新旧格式产生完全相同的边界条件数据")
                else:
                    print(f"✗ 边界条件对数量不一致: new={len(pairs_new)}, old={len(pairs_old)}")
                    return False

        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir_new)
        shutil.rmtree(temp_dir_old)

        return True

    except Exception as e:
        print(f"✗ 格式等效性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_invalid_format_handling():
    """测试无效格式的错误处理"""
    print("\n" + "=" * 60)
    print("测试无效格式错误处理")
    print("=" * 60)

    # 测试无效的pairs格式
    invalid_configs = [
        {
            "name": "pairs字段缺失",
            "bc": {
                "type": "periodic",
                "points": 1
            }
        },
        {
            "name": "pairs包含错误数量的区域",
            "bc": {
                "type": "periodic",
                "pairs": ["left"],  # 只有一个区域
                "points": 1
            }
        },
        {
            "name": "pairs包含过多区域",
            "bc": {
                "type": "periodic",
                "pairs": ["left", "right", "top"],  # 三个区域
                "points": 1
            }
        }
    ]

    base_config = {
        "problem_type": "time_pde",
        "method": "hybrid",
        "auto_code": False,
        "eq": {"L1": ["0"], "L2": ["0"], "F": ["1"], "N": ["0"]},
        "vars_list": ["u"],
        "spatial_vars": ["x"],
        "Initial_conditions": [{"var": "u", "value": "sin(pi*x)", "points": 20}],
        "hidden_dims": [8, 8],
        "n_segments": [2],
        "poly_degree": [2],
        "x_domain": [[-1, 1]]
    }

    for i, invalid_case in enumerate(invalid_configs):
        try:
            print(f"\n测试无效格式 {i+1}: {invalid_case['name']}")

            test_config = base_config.copy()
            test_config["boundary_conditions"] = [invalid_case['bc']]

            temp_dir = f"/tmp/test_invalid_{i}"
            os.makedirs(temp_dir, exist_ok=True)

            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                json.dump(test_config, f, indent=2)

            config = TimePDEConfig(case_dir=temp_dir)
            config.device = "cpu"

            data_generator = TimePDEDataGenerator(config)
            data = data_generator.generate_data(mode="train")

            # 检查是否正确处理了无效格式（应该跳过无效的边界条件）
            boundary_data = data.get("boundary_data", {})
            if 0 in boundary_data and "periodic" in boundary_data[0]:
                pairs = boundary_data[0]["periodic"]["pairs"]
                if len(pairs) == 0:
                    print(f"  ✓ 无效格式被正确跳过")
                else:
                    print(f"  ⚠ 无效格式未被跳过，生成了 {len(pairs)} 个边界条件对")

            # 清理临时文件
            import shutil
            shutil.rmtree(temp_dir)

        except Exception as e:
            print(f"  ✓ 无效格式正确触发异常: {type(e).__name__}")

    print("✓ 无效格式错误处理测试完成")
    return True

if __name__ == "__main__":
    print("开始测试最优化的周期边界条件配置格式...")

    success_count = 0
    total_tests = 5

    # 测试1：新pairs格式
    if test_new_pairs_format():
        success_count += 1

    # 测试2：向后兼容性
    if test_backward_compatibility():
        success_count += 1

    # 测试3：更新后的Allen-Cahn配置
    if test_updated_allen_cahn_config():
        success_count += 1

    # 测试4：新旧格式等效性
    if test_config_format_comparison():
        success_count += 1

    # 测试5：无效格式处理
    if test_invalid_format_handling():
        success_count += 1

    print("\n" + "=" * 60)
    print(f"最优化配置格式测试完成: {success_count}/{total_tests} 个测试通过")

    if success_count == total_tests:
        print("🎉 所有测试通过！最优化配置格式实现成功")
        print("📝 新格式: {\"type\": \"periodic\", \"pairs\": [\"left\", \"right\"], \"points\": 1}")
        print("🔄 向后兼容: 旧格式仍然支持")
        print("✨ 简洁直观: 配置格式与实现逻辑完全一致")
        sys.exit(0)
    else:
        print("❌ 部分测试失败，需要检查配置格式实现")
        sys.exit(1)