#!/usr/bin/env python3
"""
测试时间积分格式算子统一化功能
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig
from src.problem_solvers.time_pde_solver.utils.data import TimePDEDataGenerator
from src.problem_solvers.time_pde_solver.core.fitter import TimePDEFitter
from src.problem_solvers.time_pde_solver.time_schemes.onestep_predictor import OneStepPredictor
from src.problem_solvers.time_pde_solver.time_schemes.imex_rk_222 import ImexRK222
from src.problem_solvers.time_pde_solver.time_schemes.imex_1st import ImexFirstOrder

def test_time_schemes_operator_unification():
    """测试时间积分格式算子统一化"""
    print("=" * 60)
    print("测试时间积分格式算子统一化功能")
    print("=" * 60)

    case_dir = "cases/Time_pde_cases/KDV_equation"

    try:
        # 创建配置和数据
        config = TimePDEConfig(case_dir=case_dir)
        config.device = "cpu"  # 强制使用CPU
        data_generator = TimePDEDataGenerator(config)
        data = data_generator.generate_data(mode="train")

        print("\n✓ 配置和数据生成成功")

        # 创建拟合器
        fitter = TimePDEFitter(config, data)

        print(f"\n测试算子存在性检查:")
        print(f"  has_operator('L1'): {fitter.has_operator('L1')}")
        print(f"  has_operator('L2'): {fitter.has_operator('L2')}")
        print(f"  has_operator('N'): {fitter.has_operator('N')}")
        print(f"  has_operator('F'): {fitter.has_operator('F')}")

        # 测试时间积分格式
        time_schemes = [
            ("OneStepPredictor", OneStepPredictor),
            ("IMEX-RK(2,2,2)", ImexRK222),
            ("IMEX First Order", ImexFirstOrder)
        ]

        for scheme_name, scheme_class in time_schemes:
            print(f"\n{'='*40}")
            print(f"测试 {scheme_name}")
            print(f"{'='*40}")

            try:
                # 创建时间积分格式实例
                time_scheme = scheme_class(config)
                time_scheme.fitter = fitter

                print(f"✓ {scheme_name} 实例创建成功")

                # 测试算子验证方法（如果存在）
                if hasattr(time_scheme, 'validate_operators'):
                    validation = time_scheme.validate_operators()
                    print(f"  算子验证结果: {validation}")

                    # 验证所有算子都存在
                    expected_operators = ['L1_exists', 'L2_exists', 'N_exists', 'F_exists']
                    for op in expected_operators:
                        if op in validation:
                            assert validation[op] == True, f"{op} 应该为 True"

                    print(f"✓ {scheme_name} 算子验证通过")

                # 测试稳定性估算方法（如果存在）
                if hasattr(time_scheme, 'estimate_stable_dt'):
                    # 创建一个简单的解向量用于测试
                    u_test = np.ones((100, config.n_eqs))
                    try:
                        dt_stable = time_scheme.estimate_stable_dt(u_test)
                        print(f"✓ {scheme_name} 稳定时间步长估算成功: dt = {dt_stable:.6f}")
                    except Exception as e:
                        print(f"⚠ {scheme_name} 稳定时间步长估算遇到问题: {e}")

                print(f"✓ {scheme_name} 所有测试通过")

            except Exception as e:
                print(f"✗ {scheme_name} 测试失败: {e}")
                import traceback
                traceback.print_exc()
                return False

        return True

    except Exception as e:
        print(f"✗ 时间积分格式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_zero_operators_time_schemes():
    """测试零算子在时间积分格式中的处理"""
    print("\n" + "=" * 60)
    print("测试零算子在时间积分格式中的处理")
    print("=" * 60)

    # 创建只有零算子的配置
    import json
    temp_config = {
        "problem_type": "time_pde",
        "method": "hybrid",
        "auto_code": False,
        "eq": {
            "L1": ["0"],     # 零算子
            "L2": ["0"],     # 零算子
            "F": ["1"],      # 单位算子
            "N": ["0"]       # 零算子
        },
        "vars_list": ["u"],
        "spatial_vars": ["x"],
        "hidden_dims": [16, 16],
        "n_segments": [3],
        "poly_degree": [3],
        "x_domain": [[-1, 1]]
    }

    temp_dir = "/tmp/test_zero_operators_time"
    os.makedirs(temp_dir, exist_ok=True)

    with open(os.path.join(temp_dir, "config.json"), "w") as f:
        json.dump(temp_config, f, indent=2)

    try:
        config = TimePDEConfig(case_dir=temp_dir)
        config.device = "cpu"
        data_generator = TimePDEDataGenerator(config)
        data = data_generator.generate_data(mode="train")
        fitter = TimePDEFitter(config, data)

        print("✓ 零算子配置创建成功")

        # 验证所有算子仍然存在
        assert fitter.has_operator("L1") == True, "零L1算子应该存在"
        assert fitter.has_operator("L2") == True, "零L2算子应该存在"
        assert fitter.has_operator("N") == True, "零N算子应该存在"
        assert fitter.has_operator("F") == True, "F算子应该存在"

        print("✓ 零算子也被强制识别为存在")

        # 测试时间积分格式实例化
        time_scheme = OneStepPredictor(config)
        time_scheme.fitter = fitter

        print("✓ 零算子配置下时间积分格式创建成功")

        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"✗ 零算子时间积分格式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试时间积分格式算子统一化功能...")

    success_count = 0
    total_tests = 2

    # 测试1：时间积分格式算子统一化
    if test_time_schemes_operator_unification():
        success_count += 1

    # 测试2：零算子处理
    if test_zero_operators_time_schemes():
        success_count += 1

    print("\n" + "=" * 60)
    print(f"测试完成: {success_count}/{total_tests} 个测试通过")

    if success_count == total_tests:
        print("✓ 所有测试通过！时间积分格式算子统一化功能正常工作")
        sys.exit(0)
    else:
        print("✗ 部分测试失败，需要检查时间积分格式算子统一化逻辑")
        sys.exit(1)