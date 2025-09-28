#!/usr/bin/env python3
"""
测试BaseDeepPolyFitter算子强制预编译功能
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig
from src.problem_solvers.time_pde_solver.utils.data import TimePDEDataGenerator
from src.problem_solvers.time_pde_solver.core.fitter import TimePDEFitter

def test_fitter_operator_precompilation():
    """测试拟合器算子强制预编译"""
    print("=" * 60)
    print("测试 TimePDEFitter 算子强制预编译功能")
    print("=" * 60)

    case_dir = "cases/Time_pde_cases/KDV_equation"

    try:
        # 创建配置和数据
        config = TimePDEConfig(case_dir=case_dir)
        # 强制使用CPU避免设备不匹配问题
        config.device = "cpu"
        data_generator = TimePDEDataGenerator(config)
        data = data_generator.generate_data(mode="train")

        print("\n✓ 配置和数据生成成功")

        # 创建拟合器
        fitter = TimePDEFitter(config, data)

        print("\n测试算子存在性检查:")
        print(f"  has_operator('L1'): {fitter.has_operator('L1')}")
        print(f"  has_operator('L2'): {fitter.has_operator('L2')}")
        print(f"  has_operator('N'): {fitter.has_operator('N')}")
        print(f"  has_operator('F'): {fitter.has_operator('F')}")

        # 验证所有算子都返回True
        assert fitter.has_operator("L1") == True, "L1算子应该总是存在"
        assert fitter.has_operator("L2") == True, "L2算子应该总是存在"
        assert fitter.has_operator("N") == True, "N算子应该总是存在"
        assert fitter.has_operator("F") == True, "F算子应该总是存在"

        print("✓ 所有算子都强制存在")

        # 测试has_nonlinear_operators
        print(f"\n  has_nonlinear_operators(): {fitter.has_nonlinear_operators()}")
        assert fitter.has_nonlinear_operators() == True, "非线性算子检查应该总是返回True"

        print("✓ 非线性算子检查简化为总是True")

        # 进行预编译测试
        print("\n开始算子预编译...")

        # 创建一个简单的神经网络模型进行测试
        import torch
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self, hidden_dims):
                super().__init__()
                layers = []
                input_dim = len(config.spatial_vars)  # 空间维度

                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(input_dim, hidden_dim))
                    layers.append(nn.Tanh())
                    input_dim = hidden_dim

                layers.append(nn.Linear(input_dim, config.n_eqs))
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                output = self.network(x)
                return output, None  # 返回(主输出, 辅助输出)以匹配框架期望

        model = SimpleModel(config.hidden_dims)

        # 确保模型在正确的设备上，并统一dtype
        device = torch.device("cpu")  # 使用CPU避免设备不匹配
        model = model.to(device, dtype=torch.float64)  # 使用float64统一dtype

        # 测试算子强制预编译的核心逻辑，避免复杂的约束设置
        print("\n测试算子强制预编译的核心功能...")

        try:
            # 手动调用核心预编译方法（避免复杂约束问题）
            fitter._current_model = model

            # 初始化预编译状态
            fitter._precompiled = False
            fitter._linear_operators = [{} for _ in range(fitter.ns)]
            fitter._nonlinear_functions = [{} for _ in range(fitter.ns)]

            # 测试单个段的预编译（避免周期性约束问题）
            segment_idx = 0
            features = fitter._get_features(segment_idx, model)

            print(f"✓ 段 {segment_idx} 特征生成成功，特征类型: {type(features)}, 长度: {len(features) if isinstance(features, list) else 'N/A'}")

            # 验证L1算子强制编译
            L1_result = fitter.L1_func(features)
            print(f"✓ L1算子强制编译成功，返回形状: {L1_result.shape}")

            # 验证L2算子强制编译
            L2_result = fitter.L2_func(features)
            print(f"✓ L2算子强制编译成功，返回形状: {L2_result.shape}")

            print("✓ 算子强制预编译核心功能正常")

        except Exception as e:
            print(f"✗ 算子预编译核心功能测试失败: {e}")
            raise

        # 测试系统信息
        info = fitter.get_system_info()
        print(f"\n系统信息:")
        print(f"  总方程数: {info['total_equations']}")
        print(f"  总段数: {info['total_segments']}")
        print(f"  自由度: {info['degrees_of_freedom']}")
        print(f"  线性算子: {info['linear_operators']}")
        print(f"  非线性算子: {info['nonlinear_operators']}")

        # 验证信息中包含所有算子
        all_operators = info['linear_operators'] + info['nonlinear_operators']
        assert 'L1' in all_operators, "系统信息应该包含L1"
        assert 'L2' in all_operators, "系统信息应该包含L2"
        assert 'N' in all_operators, "系统信息应该包含N"
        assert 'F' in all_operators, "系统信息应该包含F"

        print("✓ 系统信息包含所有算子")

        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_zero_operator_handling():
    """测试零算子处理"""
    print("\n" + "=" * 60)
    print("测试零算子处理")
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

    temp_dir = "/tmp/test_zero_operators"
    os.makedirs(temp_dir, exist_ok=True)

    with open(os.path.join(temp_dir, "config.json"), "w") as f:
        json.dump(temp_config, f, indent=2)

    try:
        config = TimePDEConfig(case_dir=temp_dir)
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

        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"✗ 零算子测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试BaseDeepPolyFitter算子强制预编译功能...")

    success_count = 0
    total_tests = 2

    # 测试1：算子预编译
    if test_fitter_operator_precompilation():
        success_count += 1

    # 测试2：零算子处理
    if test_zero_operator_handling():
        success_count += 1

    print("\n" + "=" * 60)
    print(f"测试完成: {success_count}/{total_tests} 个测试通过")

    if success_count == total_tests:
        print("✓ 所有测试通过！算子强制预编译功能正常工作")
        sys.exit(0)
    else:
        print("✗ 部分测试失败，需要检查算子预编译逻辑")
        sys.exit(1)