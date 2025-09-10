import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

from src.problem_solvers.time_pde_solver.core.net import TimePDENet
from src.problem_solvers.time_pde_solver.core.fitter import TimePDEFitter
from src.problem_solvers.time_pde_solver.utils.data import TimePDEDataGenerator
from src.problem_solvers.time_pde_solver.utils.visualize import TimePDEVisualizer
from src.problem_solvers.time_pde_solver.utils.config import TimePDEConfig


class TimePDESolver:
    """
    优化整理后的时间PDE求解器
    支持IMEX-RK时间推进和混合神经网络-多项式方法
    """

    def __init__(self, config: Optional[TimePDEConfig] = None, case_dir: Optional[str] = None):
        """初始化求解器"""
        self._initialize_config(config, case_dir)
        self._initialize_components()
        self._initialize_solution_tracking()

    def _initialize_config(self, config: Optional[TimePDEConfig], case_dir: Optional[str]):
        """初始化配置"""
        if config is not None:
            self.config = config
        elif case_dir is not None:
            self.config = TimePDEConfig(case_dir=case_dir)
        else:
            raise ValueError("Either config object or case_dir must be provided")

        self.case_dir = case_dir if case_dir else getattr(self.config, "case_dir", None)
        
        # 导出配置文件
        if hasattr(self.config, "export_to_json"):
            self.config.export_to_json("config.json")

    def _initialize_components(self):
        """初始化核心组件"""
        # 数据生成器
        self.datagen = TimePDEDataGenerator(self.config)
        
        # 生成训练和测试数据
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")

        # 神经网络模型
        self.model = TimePDENet(self.config).to(self.config.device)
        
        # 时间PDE拟合器
        self.fitter = TimePDEFitter(config=self.config, data=self.data_train)

        # 可视化器
        self.visualizer = TimePDEVisualizer(self.config)
        
        # 创建结果目录
        os.makedirs(self.config.results_dir, exist_ok=True)

    def _initialize_solution_tracking(self):
        """初始化解跟踪"""
        self.time_history = []
        self.solution_history = []
        self.coeffs_history = []
        self.loss_history = []

    # ==================== 主求解接口 ====================

    def solve(self) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """主求解函数"""
        print("=== 开始时间PDE求解 ===")
        start_time = time.time()

        # 执行时间演化
        U_final, U_seg_final, model_final, coeffs_final = self.solve_time_evolution()

        # 计算总时间
        total_time = time.time() - start_time
        print(f"总求解时间: {total_time:.2f} 秒")

        # 后处理和可视化
        self._postprocess_results(U_final, U_seg_final, model_final, coeffs_final)

        return U_final, U_seg_final, model_final, coeffs_final

    def solve_time_evolution(self) -> Tuple[np.ndarray, list, torch.nn.Module, np.ndarray]:
        """时间演化求解 - 使用IMEX-RK方法"""
        # 初始化
        time_state = self._initialize_time_stepping()
        
        print(f"时间演化开始: T ∈ [0, {self.config.T}], dt = {self.config.dt}")
        
        # 主时间步循环
        while time_state['T'] < self.config.T:
            time_state = self._execute_time_step(time_state)
            
            # 监控和存储
            self._monitor_time_step(time_state)
            self._store_solution_history(time_state)

        print(f"时间演化完成: 最终时间 T = {time_state['T']:.6f}")
        print(f"总时间步数: {time_state['it']}, 动画帧数: {len(self.time_history)}")

        return (
            time_state['U'], 
            time_state['U_seg'], 
            self.model, 
            time_state['coeffs']
        )

    # ==================== 时间步进核心方法 ====================

    def _initialize_time_stepping(self) -> Dict:
        """初始化时间步进状态"""
        # 初始化时间状态
        time_state = {
            'it': 0,
            'T': 0.0,
            'dt': self.config.dt,
            'U': self.data_train["U"].copy(),
            'U_seg': [seg.copy() for seg in self.data_train["U_seg"]],
            'coeffs': None
        }

        # 初始化模型和拟合器
        self.fitter.fitter_init(self.model)

        # 存储初始状态
        self.time_history.append(0.0)
        U_test_initial, _ = self.fitter.construct(self.data_test, self.model, None)
        self.solution_history.append(U_test_initial.copy())

        return time_state

    def _execute_time_step(self, time_state: Dict) -> Dict:
        """执行单个时间步"""
        it = time_state['it']
        T = time_state['T']
        dt = time_state['dt']
        U = time_state['U']
        U_seg = time_state['U_seg']
        coeffs = time_state['coeffs']

        # 自适应时间步长
        dt = self._compute_adaptive_timestep(it, T, dt, U)

        print(f"步骤 {it}: T = {T:.6f}, dt = {dt:.6f}")

        # 训练神经网络
        self._train_neural_network_step(dt, U_current=U)

        # 执行IMEX时间步
        U_new, U_seg_new, coeffs_new = self.fitter.solve_time_step(
            U, U_seg, dt, coeffs_n=coeffs
        )

        # 更新状态
        return {
            'it': it + 1,
            'T': T + dt,
            'dt': dt,
            'U': U_new,
            'U_seg': U_seg_new,
            'coeffs': coeffs_new
        }

    def _compute_adaptive_timestep(self, it: int, T: float, dt: float, U: np.ndarray) -> float:
        """计算自适应时间步长"""
        base_dt = self.config.dt
        
        # 自适应时间步策略
        if hasattr(self.config, 'adaptive_dt') and self.config.adaptive_dt:
            # CFL条件限制
            if hasattr(self.config, 'cfl_number'):
                cfl_dt = self._compute_cfl_timestep(U)
                dt = min(base_dt, cfl_dt)
            
            # 稳定性限制
            if hasattr(self.fitter, "estimate_stable_dt"):
                dt_stable = self.fitter.estimate_stable_dt(U)
                dt = min(dt, dt_stable)
                
            # 解变化率限制
            if it > 0 and hasattr(self, '_previous_U'):
                dt_change = self._compute_solution_change_limit(U, self._previous_U, dt)
                dt = min(dt, dt_change)
        else:
            # 固定时间步
            dt = base_dt

        # 首步特殊处理
        if it == 0 and hasattr(self.config, 'initial_dt_factor'):
            dt *= self.config.initial_dt_factor

        # 确保不超过最终时间
        if T + dt > self.config.T:
            dt = self.config.T - T

        # 存储当前解用于下一步比较
        self._previous_U = U.copy()

        return dt
    
    def _compute_cfl_timestep(self, U: np.ndarray) -> float:
        """基于CFL条件计算时间步长"""
        # 估计特征速度 (这里需要根据具体方程调整)
        u_max = np.max(np.abs(U))
        dx_min = np.min(np.diff(self.data_train["x"].flatten()))
        
        cfl_number = getattr(self.config, 'cfl_number', 0.5)
        characteristic_speed = max(u_max, 1e-10)  # 避免除零
        
        dt_cfl = cfl_number * dx_min / characteristic_speed
        return dt_cfl
    
    def _compute_solution_change_limit(self, U_new: np.ndarray, U_old: np.ndarray, dt: float) -> float:
        """基于解变化率限制时间步"""
        rel_change = np.linalg.norm(U_new - U_old) / (np.linalg.norm(U_old) + 1e-12)
        max_change_rate = getattr(self.config, 'max_solution_change_rate', 0.1)
        
        if rel_change > max_change_rate:
            # 减小时间步
            dt_new = dt * max_change_rate / rel_change
            return max(dt_new, dt * 0.5)  # 不超过50%减少
        
        return dt

    # ==================== 神经网络训练 ====================

    def _train_neural_network_step(self, dt: float, U_current: Optional[np.ndarray] = None):
        """训练当前时间步的神经网络"""
        print("  训练神经网络...")
        
        # 准备GPU数据
        data_GPU = self.model.prepare_gpu_data(self.data_train, U_current)

        # 训练网络
        self.model.eval()
        self.model.train_net(self.data_train, self.model, data_GPU, dt=dt)
        
        # 计算最终损失
        final_loss = self.model.physics_loss(data_GPU, dt=dt).item()
        print(f"  神经网络训练完成, 损失: {final_loss:.8e}")
        
        # 更新拟合器
        self.fitter.fitter_init(self.model)

        # 记录损失历史
        self.loss_history.append(final_loss)

        # 可视化预测结果
        if self.config.debug or len(self.loss_history) % 10 == 1:
            self._create_prediction_plot(dt)

    def _create_prediction_plot(self, dt: float):
        """创建神经网络预测图"""
        try:
            print("  创建预测结果图...")
            
            # 获取测试数据和预测
            x_test = self.data_test["x"].flatten()
            x_tensor = torch.tensor(
                self.data_test["x"], dtype=torch.float64, device=self.config.device
            )
            x_tensor.requires_grad_(True)

            # 神经网络预测
            self.model.eval()
            with torch.no_grad():
                features, u_pred = self.model(x_tensor)
            
            u_pred_np = u_pred.detach().cpu().numpy().flatten()
            
            print(f"  NN预测范围: [{np.min(u_pred_np):.6f}, {np.max(u_pred_np):.6f}]")

            # 创建图形
            plt.figure(figsize=(10, 6))
            plt.scatter(x_test, u_pred_np, c="red", alpha=0.7, s=20, label="NN预测")
            
            # 如果有真实值，也绘制
            if "u" in self.data_test:
                u_true = self.data_test["u"].flatten()
                plt.scatter(x_test, u_true, c="blue", alpha=0.5, s=10, label="真实值")

            plt.xlabel("x")
            plt.ylabel("u(x)")
            plt.title(f"神经网络预测 (dt={dt:.6f})")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 保存图形
            filename = f"nn_predictions_dt_{dt:.6f}.png"
            filepath = os.path.join(self.config.results_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  神经网络预测图保存至: {filepath}")

        except Exception as e:
            print(f"  预测图创建失败: {e}")

    # ==================== 监控和存储 ====================

    def _monitor_time_step(self, time_state: Dict):
        """监控时间步进展"""
        it = time_state['it']
        T = time_state['T']
        U = time_state['U']
        
        if it % 10 == 0:
            solution_norm = np.linalg.norm(U)
            print(f"  解的范数: {solution_norm:.6e}")

    def _store_solution_history(self, time_state: Dict):
        """存储解的历史用于动画"""
        it = time_state['it']
        T = time_state['T']
        coeffs = time_state['coeffs']
        
        # 根据跳帧设置存储
        animation_skip = getattr(self.config, "animation_skip", 10)
        if it % animation_skip == 0 or T >= self.config.T:
            self.time_history.append(T)
            
            # 构建测试点处的解
            U_test, _ = self.fitter.construct(self.data_test, self.model, coeffs)
            self.solution_history.append(U_test.copy())
            
            # 存储系数
            if coeffs is not None:
                self.coeffs_history.append(coeffs.copy())

    # ==================== 后处理和可视化 ====================

    def _postprocess_results(self, U_final: np.ndarray, U_seg_final: list, 
                           model_final: torch.nn.Module, coeffs_final: np.ndarray):
        """后处理结果"""
        print("=== 后处理和可视化 ===")
        
        # 生成最终解的可视化
        test_predictions, _ = self.fitter.construct(self.data_test, model_final, coeffs_final)
        
        # 绘制最终解
        final_solution_path = os.path.join(self.config.results_dir, "final_solution.png")
        self.visualizer.plot_solution(self.data_test, test_predictions, final_solution_path)
        
        # 生成时间演化动画
        if len(self.time_history) > 1 and len(self.solution_history) > 1:
            animation_path = os.path.join(self.config.results_dir, "time_evolution.gif")
            self._create_time_evolution_animation(animation_path)
        else:
            print("警告: 没有足够的时间演化数据生成动画")

        # 生成损失曲线
        if self.loss_history:
            self._plot_loss_history()

    def _create_time_evolution_animation(self, filepath: str):
        """创建时间演化动画"""
        try:
            print(f"创建时间演化动画: {filepath}")
            
            # 准备动画数据
            animation_data = {
                'times': self.time_history,
                'solutions': self.solution_history,
                'x_coords': self.data_test["x"].flatten()
            }
            
            # 使用可视化器创建动画
            self.visualizer.create_time_animation(animation_data, filepath, duration=200)
            print(f"动画保存完成: {filepath}")
            
        except Exception as e:
            print(f"动画创建失败: {e}")

    def _plot_loss_history(self):
        """绘制损失历史"""
        try:
            plt.figure(figsize=(10, 6))
            plt.semilogy(self.loss_history, 'b-', linewidth=2)
            plt.xlabel('时间步')
            plt.ylabel('训练损失 (对数尺度)')
            plt.title('神经网络训练损失演化')
            plt.grid(True, alpha=0.3)
            
            filepath = os.path.join(self.config.results_dir, "loss_history.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"损失历史图保存至: {filepath}")
            
        except Exception as e:
            print(f"损失历史图创建失败: {e}")

    # ==================== 便利接口 ====================

    def get_solution_at_time(self, T_target: float) -> Tuple[np.ndarray, np.ndarray]:
        """获取指定时间的解"""
        if not self.time_history:
            raise RuntimeError("需要先运行solve()方法")
            
        # 找到最接近的时间点
        time_array = np.array(self.time_history)
        idx = np.argmin(np.abs(time_array - T_target))
        
        return self.solution_history[idx], time_array[idx]

    def get_convergence_info(self) -> Dict:
        """获取收敛信息"""
        if not self.loss_history:
            return {"error": "没有可用的收敛数据"}
            
        return {
            "final_loss": self.loss_history[-1],
            "initial_loss": self.loss_history[0],
            "reduction_ratio": self.loss_history[0] / self.loss_history[-1],
            "total_steps": len(self.loss_history),
            "average_loss": np.mean(self.loss_history)
        }

    def export_solution_data(self, filepath: str):
        """导出解数据"""
        export_data = {
            'config': self.config.__dict__,
            'time_history': self.time_history,
            'solution_history': self.solution_history,
            'loss_history': self.loss_history,
            'x_coords': self.data_test["x"].flatten()
        }
        
        np.savez_compressed(filepath, **export_data)
        print(f"解数据导出至: {filepath}")


# ==================== 便利函数 ====================

def solve_time_pde(case_dir: str, **kwargs) -> TimePDESolver:
    """便利函数：直接求解时间PDE"""
    solver = TimePDESolver(case_dir=case_dir)
    solver.solve()
    return solver


def create_solver_from_config(config_dict: Dict) -> TimePDESolver:
    """从配置字典创建求解器"""
    config = TimePDEConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    return TimePDESolver(config=config)