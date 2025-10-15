"""
NonlinearPDESolver - 统一的非线性PDE求解器基类

基于统一迭代框架设计，支持三种求解策略：
1. TimePDESolver - 物理时间演化
2. GeneralPDESolver - 伪时间步迭代至稳态
3. FixedPointSolver - 不动点迭代（无时间步）

核心思想：所有求解都是迭代 U^{n+1} = F(U^n, state)
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import torch
import scipy.io
import os


@dataclass
class IterationState:
    """迭代状态 - 统一的状态管理

    用于管理三种不同求解策略的迭代状态：
    - TimePDE: 使用 iter, time, dt
    - GeneralPDE: 使用 iter, time (τ), dt (dτ), residual, converged
    - FixedPoint: 使用 iter, residual, converged
    """
    # 共性状态：迭代次数
    iter: int = 0

    # 时间相关（TimePDE, GeneralPDE）
    time: float = 0.0                # 当前时间（T 或 τ）
    dt: float = 0.0                  # 时间步长

    # 收敛相关（GeneralPDE, FixedPoint）
    residual: float = float('inf')   # 残差
    converged: bool = False          # 是否收敛

    # 解相关
    U_prev: Optional[np.ndarray] = None  # 上一步的解
    coeffs: Optional[np.ndarray] = None  # 当前系数

    # 其他自定义状态
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """确保 custom 字典已初始化"""
        if self.custom is None:
            self.custom = {}


class NonlinearPDESolver(ABC):
    """非线性PDE求解器统一基类 - 基于统一迭代框架

    核心思想：所有求解策略都是迭代 U^{n+1} = F(U^n, state)

    区别仅在于：
    1. 终止条件：时间达到 vs 残差收敛
    2. 迭代步骤：time_step vs fit
    3. 状态更新：时间递增 vs 残差计算

    代码复用率：~80%
    """

    # ==================== 统一迭代框架（核心方法） ====================

    def _unified_iteration_loop(
        self,
        U_init: np.ndarray,
        initial_state: IterationState,
        iteration_step: Callable[[np.ndarray, IterationState], Tuple[np.ndarray, IterationState]],
        termination_check: Callable[[IterationState], bool],
        state_update: Optional[Callable[[np.ndarray, np.ndarray, IterationState], IterationState]] = None,
        print_progress: Optional[Callable[[IterationState], None]] = None
    ) -> Tuple[np.ndarray, IterationState]:
        """统一的迭代循环框架 - 三种 Solver 的共同抽象

        这是最核心的方法，提供了一个通用的迭代框架。

        Args:
            U_init: 初始解 (n_points, n_eqs)
            initial_state: 初始迭代状态
            iteration_step: 迭代步骤函数
                签名：(U, state) -> (U_new, state_new)
                - TimePDE: time_step(U, state) -> (U_new, state)
                - GeneralPDE: pseudo_time_step(U, state) -> (U_new, state)
                - FixedPoint: fixed_point_step(U, state) -> (U_new, state)

            termination_check: 终止条件检查函数
                签名：(state) -> bool
                - TimePDE: lambda s: s.time >= T_final
                - GeneralPDE: lambda s: s.converged or s.iter >= max_iter
                - FixedPoint: lambda s: s.converged or s.iter >= max_iter

            state_update: 状态更新函数（可选）
                签名：(U_old, U_new, state) -> state_new
                用于自定义状态更新逻辑（如自适应时间步）

            print_progress: 进度打印函数（可选）
                签名：(state) -> None

        Returns:
            (U_final, final_state): 最终解和最终状态
        """
        U = U_init.copy()
        state = initial_state

        while not termination_check(state):
            # 执行一步迭代
            U_new, state_new = iteration_step(U, state)

            # 可选：自定义状态更新（如自适应步长）
            if state_update is not None:
                state_new = state_update(U, U_new, state_new)

            # 更新解和状态
            U = U_new
            state = state_new

            # 可选：打印进度
            if print_progress is not None:
                print_progress(state)

        return U, state

    # ==================== 辅助方法（具体方法 - 100%共性） ====================

    def _compute_residual(self, U_new: np.ndarray, U_old: np.ndarray) -> float:
        """计算相对残差 - 标准实现

        用于 GeneralPDE 和 FixedPoint 的收敛判断

        Args:
            U_new: 新解
            U_old: 旧解

        Returns:
            相对残差 ||U_new - U_old|| / ||U_old||
        """
        return np.linalg.norm(U_new - U_old) / (np.linalg.norm(U_old) + 1e-10)

    def _update_convergence_state(
        self,
        U_old: np.ndarray,
        U_new: np.ndarray,
        state: IterationState,
        conv_tol: float
    ) -> IterationState:
        """更新收敛状态 - 标准实现

        用于 GeneralPDE 和 FixedPoint

        Args:
            U_old: 旧解
            U_new: 新解
            state: 当前状态
            conv_tol: 收敛容差

        Returns:
            更新后的状态
        """
        residual = self._compute_residual(U_new, U_old)
        state.residual = residual
        state.converged = (residual < conv_tol)
        state.U_prev = U_old.copy()
        state.iter += 1
        return state

    def _update_time_state(
        self,
        state: IterationState,
        dt: Optional[float] = None
    ) -> IterationState:
        """更新时间状态 - 标准实现

        用于 TimePDE 和 GeneralPDE

        Args:
            state: 当前状态
            dt: 时间步长（如果为 None，使用 state.dt）

        Returns:
            更新后的状态
        """
        if dt is None:
            dt = state.dt
        state.time += dt
        state.dt = dt
        state.iter += 1
        return state

    def _train_neural_network_step(
        self,
        iter: int,
        dt: float,
        U_current: np.ndarray,
        force: bool = False
    ):
        """神经网络训练步骤 - 通用框架

        根据配置决定是否训练神经网络，并更新 fitter

        Args:
            iter: 当前迭代次数
            dt: 当前时间步长
            U_current: 当前解
            force: 是否强制训练（忽略 spotter_skip）
        """
        # 记录最近一次训练损失（用于子类钩子）
        self._last_training_loss: Optional[float] = None

        # 检查是否需要跳过神经网络训练
        if not force and not self._should_train_network(iter):
            # 仍然需要重新初始化 fitter，但使用现有模型参数
            self.fitter.fitter_init(self.model)
            return

        print(f"  Training Neural Network...")

        # 准备 GPU 数据
        data_GPU = self.model.prepare_gpu_data(self.data_train, U_current)

        # 训练模式
        self.model.train()

        # 执行训练
        self.model.train_net(self.data_train, self.model, data_GPU, dt=dt)

        # 评估模式并计算最终损失
        self.model.eval()
        final_loss = self.model.physics_loss(data_GPU, dt=dt).item()
        print(f"  ✓ Training Completed - Final Loss: {final_loss:.6e}")

        # 记录最后一次训练损失，供子类使用
        self._last_training_loss = final_loss

        # 重新初始化 fitter 以使用更新后的模型
        self.fitter.fitter_init(self.model)

        # 触发子类钩子以便执行额外的后处理（例如记录、可视化）
        self._on_network_trained(iter, dt, U_current, final_loss)

    def _should_train_network(self, iter: int) -> bool:
        """判断是否需要训练网络 - 可覆盖

        Args:
            iter: 当前迭代次数

        Returns:
            是否需要训练
        """
        spotter_skip = getattr(self.config, 'spotter_skip', 1)
        return (iter % spotter_skip == 0)

    # ==================== 初始化方法（100%共性） ====================

    def __init__(self, config: Optional[Any] = None, case_dir: Optional[str] = None) -> None:
        """统一的初始化流程

        Args:
            config: 配置对象（可选）
            case_dir: 案例目录路径（可选）
        """
        self._initialize_config(config, case_dir)
        self._initialize_components()
        self._solver_specific_init()

    def _initialize_config(self, config: Optional[Any], case_dir: Optional[str]) -> None:
        """配置初始化 - 100%相同

        Args:
            config: 配置对象
            case_dir: 案例目录
        """
        if config is not None:
            self.config = config
        elif case_dir is not None:
            self.config = self._create_config(case_dir)
        else:
            raise ValueError("Either config object or case_dir must be provided")

        self.case_dir = case_dir if case_dir else getattr(self.config, "case_dir", None)

        # 导出配置文件 - 已禁用，避免覆盖原始config.json
        # if hasattr(self.config, "export_to_json"):
        #     self.config.export_to_json("config.json")

    def _initialize_components(self) -> None:
        """组件初始化 - 100%相同

        创建并初始化所有核心组件：
        - 数据生成器
        - 训练和测试数据
        - 神经网络模型
        - 可视化器
        - Fitter
        - 参考解
        """
        # 数据生成器
        self.datagen = self._create_data_generator()

        # 生成训练和测试数据
        self.data_train = self.datagen.generate_data("train")
        self.data_test = self.datagen.generate_data("test")

        # 神经网络模型
        self.model = self._create_model().to(self.config.device)

        # 可视化器
        self.visualizer = self._create_visualizer()

        # Fitter（使用默认或配置的时间格式）
        time_scheme = getattr(self.config, 'time_scheme', self._get_default_time_scheme())
        self.fitter = self._create_fitter(time_scheme)

        # 加载参考解（如果指定）
        self.reference_solution = self._load_reference_solution()

        # 创建结果目录
        os.makedirs(self.config.results_dir, exist_ok=True)

    def _load_reference_solution(self) -> Optional[Dict]:
        """加载参考解 - 85%相同，具体处理由子类实现

        Returns:
            参考解字典，如果不存在则返回 None
        """
        if not hasattr(self.config, 'reference_solution') or not self.config.reference_solution:
            return None

        ref_path = self.config.reference_solution

        # 如果是相对路径，转为绝对路径
        if not os.path.isabs(ref_path):
            case_dir = getattr(self.config, 'case_dir', os.getcwd())
            ref_path = os.path.join(case_dir, ref_path)

        # 检查 .mat 文件
        if ref_path.endswith('.m'):
            mat_file = ref_path.replace('.m', '.mat')
        else:
            mat_file = ref_path

        if not os.path.exists(mat_file):
            print(f"Warning: Reference solution file not found: {mat_file}")
            return None

        try:
            print(f"Loading reference solution: {mat_file}")
            data = scipy.io.loadmat(mat_file)

            # 提取 x 坐标（所有求解器都需要）
            if 'x' in data:
                x_ref = data['x'].flatten()
            else:
                print("Warning: 'x' not found in reference solution file")
                return None

            # 委托给子类处理具体数据
            return self._process_reference_solution(data, x_ref, mat_file)

        except Exception as e:
            print(f"Error: Cannot read reference solution file {mat_file}: {e}")
            return None

    def _to_numpy_cpu(self, data):
        """GPU数据转CPU numpy数组 - 工具方法

        Args:
            data: 输入数据（torch.Tensor 或 np.ndarray）

        Returns:
            numpy 数组，如果转换失败则返回原始数据
        """
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        if isinstance(data, np.ndarray):
            return np.asarray(data)
        try:
            return np.asarray(data)
        except:
            return data

    def _interpolate_to_test_grid(self, U_train: np.ndarray) -> np.ndarray:
        """从训练点插值到测试网格 - 工具方法

        Args:
            U_train: 训练点上的解 (n_train, n_eqs)

        Returns:
            测试点上的解 (n_test, n_eqs)
        """
        from scipy.interpolate import griddata

        x_train = self.data_train['x']
        x_test = self.data_test['x']
        n_eqs = U_train.shape[1] if U_train.ndim > 1 else 1

        if n_eqs == 1:
            U_train = U_train.reshape(-1, 1)

        U_test = np.zeros((x_test.shape[0], n_eqs))

        for eq_idx in range(n_eqs):
            # 线性插值
            result = griddata(
                x_train, U_train[:, eq_idx], x_test,
                method='linear', fill_value=np.nan
            )
            U_test[:, eq_idx] = result.flatten() if result.ndim > 1 else result

            # 对 NaN 值使用最近邻插值
            nan_mask = np.isnan(U_test[:, eq_idx])
            if np.any(nan_mask):
                result_nn = griddata(
                    x_train, U_train[:, eq_idx], x_test[nan_mask],
                    method='nearest'
                )
                U_test[nan_mask, eq_idx] = result_nn.flatten() if result_nn.ndim > 1 else result_nn

        print(f"  Interpolated from {x_train.shape[0]} training points to {x_test.shape[0]} test points")
        return U_test

    # ==================== 抽象方法（子类必须实现） ====================

    @abstractmethod
    def solve(self, **kwargs) -> Dict[str, Any]:
        """主求解方法 - 子类使用 _unified_iteration_loop 实现

        子类应该：
        1. 定义 iteration_step 函数
        2. 定义 termination_check 函数
        3. 可选定义 state_update 和 print_progress 函数
        4. 调用 _unified_iteration_loop
        5. 返回结果字典
        """
        pass

    @abstractmethod
    def _create_config(self, case_dir: str):
        """创建配置对象 - 工厂方法

        Args:
            case_dir: 案例目录

        Returns:
            配置对象
        """
        pass

    @abstractmethod
    def _create_data_generator(self):
        """创建数据生成器 - 工厂方法

        Returns:
            数据生成器对象
        """
        pass

    @abstractmethod
    def _create_model(self):
        """创建神经网络模型 - 工厂方法

        Returns:
            模型对象
        """
        pass

    @abstractmethod
    def _create_fitter(self, time_scheme: Optional[str]):
        """创建Fitter对象 - 工厂方法

        Args:
            time_scheme: 时间格式（可选）

        Returns:
            Fitter 对象
        """
        pass

    @abstractmethod
    def _create_visualizer(self):
        """创建可视化器 - 工厂方法

        Returns:
            可视化器对象
        """
        pass

    @abstractmethod
    def _get_default_time_scheme(self) -> Optional[str]:
        """获取默认时间格式 - 工厂方法

        Returns:
            默认时间格式字符串，如果不适用则返回 None
        """
        pass

    @abstractmethod
    def _process_reference_solution(self, data: Dict, x_ref: np.ndarray, mat_file: str) -> Dict:
        """处理参考解数据 - 维度特定处理

        不同维度的 PDE 有不同的数据结构：
        - 1D+时间: (x, t, u)
        - 2D稳态: (x, y, u)

        Args:
            data: MATLAB 数据字典
            x_ref: x 坐标
            mat_file: 文件路径

        Returns:
            处理后的参考解字典
        """
        pass

    # ==================== 钩子方法（可选覆盖） ====================

    def _solver_specific_init(self) -> None:
        """求解器特定初始化 - 钩子方法

        子类可以覆盖此方法进行特定初始化，如：
        - TimePDE: 初始化 loss_history 和动画数据
        - GeneralPDE: 初始化收敛历史
        - FixedPoint: 初始化迭代历史
        """
        pass

    def _on_network_trained(
        self,
        iter: int,
        dt: float,
        U_current: np.ndarray,
        final_loss: float
    ) -> None:
        """神经网络训练完成后的钩子 - 子类可覆盖

        默认实现为空。TimePDE 等求解器可用来记录 loss_history、触发可视化等。

        Args:
            iter: 当前迭代次数
            dt: 当前时间步长
            U_current: 当前解（训练输入）
            final_loss: 训练后计算得到的物理损失
        """
        # 默认不执行任何操作，由子类根据需要覆盖
        _ = (iter, dt, U_current, final_loss)

    def _postprocess_results(self, **kwargs: Any) -> None:
        """后处理和可视化 - 钩子方法

        子类可以覆盖此方法进行特定后处理

        Args:
            **kwargs: 子类特定的参数
        """
        # 钩子方法，参数由子类使用
        _ = kwargs

    def _get_initial_guess(self) -> np.ndarray:
        """获取初始猜测值 - 钩子方法

        默认实现：零初值

        Returns:
            初始猜测值 (n_points, n_eqs)
        """
        n_points = self.data_train['x'].shape[0]
        n_eqs = self.config.n_eqs
        return np.zeros((n_points, n_eqs))

    def _initialize_from_config(self) -> np.ndarray:
        """根据配置生成初始猜测值 - 通用方法

        支持三种初始化方法：
        1. 'zero': 零初值
        2. 'spotter': 使用Spotter网络预训练
        3. 'given': 从Initial_conditions表达式计算

        Returns:
            初始猜测值 (n_points, n_eqs)
        """
        # 获取初始化方法配置
        initial_condition = getattr(self.config, 'initial_condition', {})
        if isinstance(initial_condition, dict):
            init_method = initial_condition.get('method', 'zero')
        else:
            init_method = 'zero'

        print(f"Initializing solution using method: '{init_method}'")

        if init_method == 'zero':
            return self._get_initial_guess()
        elif init_method == 'spotter':
            return self._spotter_initialization()
        elif init_method == 'given':
            return self._config_based_initialization()
        else:
            raise ValueError(f"Unknown initial_condition method: {init_method}")

    def _spotter_initialization(self) -> np.ndarray:
        """使用Spotter网络预训练初始化 - 通用方法

        通用流程：
        1. 创建SpotterNet（使用与主网络相同的架构）
        2. 训练Spotter（最小化物理损失）
        3. 转移权重到主网络
        4. 提取预测值作为初值

        Returns:
            从Spotter网络预测的初始解 (n_points, n_eqs)

        Note:
            子类需要确保有SpotterNet类可用，或覆盖此方法
        """
        print("  Using Spotter network for initialization...")

        # 获取Spotter配置
        spotter_config = getattr(self.config, 'initial_condition', {}).get('spotter_config', {})
        n_epochs = spotter_config.get('epochs', 1000)
        lr_init = spotter_config.get('learning_rate', 1e-2)
        n_pde_points = spotter_config.get('n_pde_points', 10000)
        print_interval = spotter_config.get('print_interval', 200)

        # 尝试导入SpotterNet
        try:
            # 子类特定的SpotterNet导入
            SpotterNet = self._get_spotter_net_class()
        except (NotImplementedError, ImportError) as e:
            print(f"  Warning: SpotterNet not available ({e})")
            print("  Falling back to zero initialization")
            return self._get_initial_guess()

        # 创建并训练Spotter
        spotter_model = SpotterNet(self.config, use_periodic=False).to(self.config.device)

        print(f"  Training SpotterNet with {n_epochs} epochs...")
        print(f"  PDE sampling points: {n_pde_points}")
        print(f"  Initial learning rate: {lr_init}")

        spotter_model.train_spotter(
            data=self.data_train,
            n_epochs=n_epochs,
            lr_init=lr_init,
            n_pde_points=n_pde_points,
            print_every=print_interval
        )

        # 转移权重到主网络
        print("\n  Transferring network weights from SpotterNet to main network...")
        self.model.load_state_dict(spotter_model.state_dict())
        print("  ✓ Network weights transferred successfully")

        # 提取初始解
        with torch.no_grad():
            x_train_tensor = torch.tensor(
                self.data_train['x'], dtype=torch.float64,
                device=self.config.device, requires_grad=True
            )
            _, U_pred = self.model(x_train_tensor)
            U_init = U_pred.cpu().numpy()

            u_norm = np.linalg.norm(U_init)
            u_max = np.max(np.abs(U_init))
            print(f"  Initial solution: ||U|| = {u_norm:.6e}, max|U| = {u_max:.6e}")

        return U_init

    def _config_based_initialization(self) -> np.ndarray:
        """根据Initial_conditions表达式计算初值 - 通用方法

        从config.Initial_conditions读取表达式，如：
        ```yaml
        Initial_conditions:
          - var: u
            value: "sin(2*pi*x[:, 0])"
          - var: v
            value: "cos(2*pi*x[:, 1])"
        ```

        Returns:
            从表达式计算的初始解 (n_points, n_eqs)
        """
        if not hasattr(self.config, 'Initial_conditions'):
            raise ValueError("No Initial_conditions found in config for method='given'")

        print("  Using given initial conditions from config...")

        n_points = self.data_train['x'].shape[0]
        n_eqs = self.config.n_eqs
        U_init = np.zeros((n_points, n_eqs))

        # 获取空间坐标
        x_coords = self.data_train['x']

        # 处理每个初始条件
        for ic in self.config.Initial_conditions:
            var_name = ic.get('var')
            value_expr = ic.get('value')

            # 找到变量索引
            if not hasattr(self.config, 'vars_list'):
                # 如果没有vars_list，假设变量按顺序排列
                var_idx = int(var_name.replace('u', '').replace('v', '').replace('w', '') or '0')
            else:
                var_idx = self.config.vars_list.index(var_name)

            # 构建命名空间用于eval
            namespace = {
                'np': np,
                'pi': np.pi,
                'cos': np.cos,
                'sin': np.sin,
                'exp': np.exp,
                'sqrt': np.sqrt,
                'tanh': np.tanh,
                'x': x_coords  # 提供完整的x数组
            }

            # 如果有空间变量名称列表，添加每个维度
            if hasattr(self.config, 'spatial_vars'):
                for i, var in enumerate(self.config.spatial_vars):
                    namespace[var] = x_coords[:, i]

            # 计算表达式
            try:
                U_init[:, var_idx] = eval(value_expr, namespace)
                print(f"  Initialized {var_name} with: {value_expr}")

                u_norm = np.linalg.norm(U_init[:, var_idx])
                u_max = np.max(np.abs(U_init[:, var_idx]))
                print(f"    ||{var_name}|| = {u_norm:.6e}, max|{var_name}| = {u_max:.6e}")
            except Exception as e:
                print(f"  Warning: Failed to evaluate expression for {var_name}: {e}")
                print(f"  Using zero initialization for {var_name}")

        return U_init

    def _get_spotter_net_class(self):
        """获取SpotterNet类 - 钩子方法

        子类应该覆盖此方法以提供自己的SpotterNet实现

        Returns:
            SpotterNet类（未实例化）

        Raises:
            NotImplementedError: 如果子类没有提供SpotterNet
        """
        raise NotImplementedError("Subclass must provide SpotterNet class or override _spotter_initialization()")

    def _adapt_pseudo_timestep(self, dt: float, residual: float) -> float:
        """自适应伪时间步调整 - 钩子方法

        默认实现：不调整（返回原始dt）
        GeneralPDESolver 应该覆盖此方法实现自适应策略

        Args:
            dt: 当前时间步长
            residual: 当前残差

        Returns:
            调整后的时间步长
        """
        return dt

    # ==================== 时间步控制方法（新增 - 从Fitter移到Solver层） ====================

    def validate_operator_configuration(self) -> Dict[str, bool]:
        """验证算子配置 - 通用方法

        检查fitter的time_scheme和算子配置是否有效。
        所有solver都可以使用此方法验证配置。

        Returns:
            验证结果字典，包含验证状态和可能的错误信息
        """
        if not hasattr(self.fitter, 'time_scheme') or self.fitter.time_scheme is None:
            return {
                "valid": False,
                "error": "No time scheme initialized"
            }

        return self.fitter.time_scheme.validate_operators()

    def get_solver_time_info(self) -> Dict[str, Any]:
        """获取求解器时间信息 - 基础接口（子类覆盖）

        返回求解器的时间步相关信息。子类应该覆盖此方法
        提供特定的时间信息。

        Returns:
            时间信息字典，包含solver类型和时间步配置等
        """
        return {
            "solver_type": self.__class__.__name__,
            "has_time_scheme": hasattr(self.fitter, 'time_scheme') and self.fitter.time_scheme is not None
        }
