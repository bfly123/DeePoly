import numpy as np
import torch
import time
from typing import Dict, Tuple
from torch import optim
import os
import argparse
import sys

# 确保可以导入项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 将相对导入改为绝对导入
from src.problem_solvers.time_pde_solver.core.net import TimePDENet
# 导入train函数
#from src.abstract_class.base_net import train_net
from src.problem_solvers.time_pde_solver.core.fitter import TimePDEFitter
from src.problem_solvers.time_pde_solver.utils.data import TimePDEDataGenerator
from src.problem_solvers.time_pde_solver.utils.visualize import TimePDEVisualizer
from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig


class TimePDESolver:
    def __init__(self, config):
        # 初始化配置
        self.config = config
        self.config.export_to_json("config.json")

        # 初始化数据生成器
        self.datagen = TimePDEDataGenerator(self.config)

        # 准备训练和测试数据
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")

        # 初始化模型
        self.model = TimePDENet(
            in_dim=2, hidden_dims=self.config.hidden_dims, out_dim=self.config.n_eqs
        ).to(self.config.device)

        # 准备GPU数据
        self.data_GPU = self.datagen.prepare_gpu_data(self.data_train)

        # 初始化拟合器
        self.fitter = TimePDEFitter(config=self.config, data=self.data_train)

        # 创建结果目录
        os.makedirs("./Case2/results/evolution", exist_ok=True)

        # 初始化可视化器
        self.visualizer = TimePDEVisualizer(self.config)

    def solve(self) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """主求解函数"""
        start_time = time.time()

        u_n, u_n_seg, model, coeffs = self.solve_time_evolution()

        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f} seconds")

        # 评估结果
        test_predictions, _ = self.fitter.construct(self.data_test, model, coeffs)
        self.visualizer.plot_solution(self.data_test, test_predictions, "results/final_solution.png")

        # 保存动画
        self.visualizer.save_animation("results/flow_evolution.gif", duration=200)

        return u_n, u_n_seg, model, coeffs

    def solve_time_evolution(
        self,
    ) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """时间演化求解，使用TR-BDF2格式"""
        # 初始化时间步参数
        it = 0
        T = 0
        dt = self.config.dt

        # 初始化解和分段解
        u_n = self.data_train["u"]
        u_n_seg = self.data_train["u_segments"]
        f_n = None  # 存储空间导数项

        while T < self.config.time:
            # 调整时间步长
            if it == 0:
                dt = self.config.dt / 10
            else:
                dt = self.config.dt

            it += 1
            T += dt

            if T + dt > self.config.time:
                dt = self.config.time - T
            print(f"T = {T:.3f}")

            if it == 1:
                # 第一步使用一阶方法
                u_np1, u_np1_seg, f_np1, f_np1_seg, self.model, coeffs = (
                    self._time_evolve(
                        "1st_order",
                        {
                            "u_n": u_n,
                            "u_ng": None,
                            "f_n": None,
                        },{
                            "u_n_seg": u_n_seg,
                            "u_ng_seg": None,
                            "f_n_seg": None,
                        },
                        dt,
                    )
                )
            else:
                # TR-BDF2第一阶段：计算u^{n+γ}
                u_np1, u_np1_seg, f_np1, f_np1_seg, self.model, coeffs = (
                    self._time_evolve(
                        "pre",
                        {
                            "u_n": u_n,
                            "u_ng": None,
                            "f_n": f_n,
                        },
                        {
                            "u_n_seg": u_n_seg,
                            "u_ng_seg": None,
                            "f_n_seg": f_n_seg,
                        },
                        dt,
                    )
                )

            # 更新解
            u_n = u_np1
            u_n_seg = u_np1_seg
            f_n = f_np1
            f_n_seg = f_np1_seg
            
            # 可视化当前时间步
            self.visualizer.plot_evolution_step(
                T, u_n, self.data_train["x"],
                f"Case2/results/evolution/t_{T:.3f}.png"
            )

        self.visualizer.close_evolution_plot()
        return u_n, u_n_seg, self.model, coeffs

    def _time_evolve(
        self, step: str, data: Dict, data_seg: Dict, dt: float
    ) -> Tuple[np.ndarray, list, np.ndarray, list, torch.nn.Module, np.ndarray]:
        """单个时间步的演化"""
        # 训练模型
        train(data, self.model, self.data_GPU, self.config, optim, step=step, dt=dt)
        self.fitter.fitter_init(self.model)

        # 拟合并预测
        coeffs = self.fitter.fit(data_seg, step=step, dt=dt)
        u, u_seg = self.fitter.construct(self.data_train, self.model, coeffs)

        # 计算空间导数
        u_x, u_x_seg = self.fitter.construct(
            self.data_train, self.model, coeffs, [1, 0]
        )
        u_y, u_y_seg = self.fitter.construct(
            self.data_train, self.model, coeffs, [0, 1]
        )

        # 计算总的空间导数项
        f = u_x + u_y
        f_seg = [x_seg + y_seg for x_seg, y_seg in zip(u_x_seg, u_y_seg)]

        return u, u_seg, f, f_seg, self.model, coeffs


def main():
    parser = argparse.ArgumentParser(description='求解器入口')
    parser.add_argument('--case', type=str, default='time_dependent',
                      choices=['time_dependent'],
                      help='选择要运行的算例')
    args = parser.parse_args()

    if args.case == 'time_dependent':
        config = TimePDEConfig()
        solver = TimePDESolver(config)
        solver.solve()
    else:
        raise ValueError(f"未知的算例类型: {args.case}")


if __name__ == "__main__":
    main() 