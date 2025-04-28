from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import os
import json
import importlib.util
import argparse
import time
import sys

# 利用__init__.py简化导入
from .core import FuncFittingNet, FuncFittingFitter
from .utils import FuncFittingConfig, FuncFittingDataGenerator

class FuncFittingSolver:
    """函数拟合求解器"""

    def __init__(self, config=None, case_dir=None):
        # 保存案例路径
        self.case_dir = case_dir
        self.config = config
        # 准备数据生成器和数据
        self.datagen = FuncFittingDataGenerator(self.config)
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")

        # 初始化模型
        self.model = FuncFittingNet(self.config).to(self.config.device)


        # 加载统一的plot模块
        self.output_module = self._load_output_module()

    def _load_output_module(self):
        """加载案例目录中的output.py模块，该模块同时处理输出和可视化"""
        if not self.case_dir:
            raise ValueError("案例路径未设置，无法加载output.py模块")

        output_path = os.path.join(self.case_dir, "output.py")

        if not os.path.exists(output_path):
            raise ValueError(f"未找到必需的output.py模块: {output_path}")

        try:
            print(f"加载自定义输出和可视化模块: {output_path}")
            # 将案例目录添加到Python路径
            if self.case_dir not in sys.path:
                sys.path.insert(0, self.case_dir)
            
            spec = importlib.util.spec_from_file_location(
                "custom_output_module", output_path
            )
            output_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(output_module)
            return output_module
        except Exception as e:
            raise RuntimeError(f"加载output.py模块时出错: {e}")

    def solve(self, result_dir=None):
        """
        求解函数拟合问题

        Args:
            result_dir: 结果保存目录，如果为None则使用当前目录下的results

        Returns:
            tuple: (train_predictions, test_predictions) 训练和测试集的预测结果
        """
        print("开始求解函数拟合问题...")
        
        # 记录总求解时间
        solve_start_time = time.time()

        # 确定结果保存目录
        if result_dir is None:
            result_dir = os.path.join(self.case_dir, "results")
        os.makedirs(result_dir, exist_ok=True)
        print(f"结果将保存在: {os.path.abspath(result_dir)}")

        # 初始化并执行拟合
        print("初始化拟合器...")

        self.fitter = FuncFittingFitter(self.config, self.data_train)
        data_GPU = self.model.prepare_gpu_data(self.data_train)
        
        # 训练网络并记录最终损失
        print("开始训练神经网络...")
        self.model = self.model.train_net(self.data_train, self.model, data_GPU)
        
        # 获取最终损失值（在评估模式下计算一次）
        self.model.eval()
        with torch.no_grad():
            final_loss = self.model.physics_loss(data_GPU).item()
        self.model.train()
        print(f"神经网络训练完成，最终损失: {final_loss:.8e}")
        
        self.fitter.fitter_init(self.model)

        print("开始拟合数据...")
        start_time = time.time()
        coeffs = self.fitter.fit()
        fit_time = time.time() - start_time
        print(f"拟合完成，耗时: {fit_time:.2f}秒")

        
        # 计算总求解时间
        total_solve_time = time.time() - solve_start_time
        print(f"求解总耗时: {total_solve_time:.2f}秒")


        # 进行预测
        print("在训练集上进行预测...")
        train_predictions, train_segments = self.fitter.construct(
            self.data_train, self.model, coeffs
        )

        print("在测试集上进行预测...")
        test_predictions, test_segments = self.fitter.construct(
            self.data_test, self.model, coeffs
        )
        # 使用统一的plot模块处理结果输出和可视化
        print("使用plot模块生成结果...")
        output_data = {
            "train_data": self.data_train,
            "test_data": self.data_test,
            "train_predictions": train_predictions,
            "test_predictions": test_predictions,
            "train_segments": train_segments,
            "test_segments": test_segments,
            "coeffs": coeffs,
            "model": self.model,
            "config": self.config,
            "result_dir": result_dir,
            "final_loss": final_loss,         # 添加最终损失
            "solution_time": total_solve_time  # 添加总求解时间
        }
        self.output_module.generate_output(output_data)

        print("函数拟合求解完成!")
        return train_predictions, test_predictions


def main():
    """命令行入口函数 - 仅用于直接测试"""
    # 确保当前路径在Python导入路径中
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 指定固定的案例路径
    case_dir = "cases/func_fitting_cases/test_sin"
    print(f"使用指定案例路径: {case_dir}")

    # 构造配置文件完整路径
    config_path = os.path.join(case_dir, "config.json")

    if os.path.exists(config_path):
        # 保存当前工作目录
        original_dir = os.getcwd()

        try:
            # 初始化并运行求解器
            solver = FuncFittingSolver(case_path=case_dir)
            solver.solve()
        finally:
            # 恢复原始工作目录
            os.chdir(original_dir)
    else:
        raise ValueError(f"配置文件未找到: {config_path}")


if __name__ == "__main__":
    # 设置__package__变量，解决直接运行时的相对导入问题
    package_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(package_path, "../../..")))
    __package__ = "src.problem_solvers.func_fitting_solver"

    main()
