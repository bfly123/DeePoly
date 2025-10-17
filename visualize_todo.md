# 可视化系统统一改进计划 (方案A - 渐进式)

## 项目目标
统一DeePoly项目中所有求解器的可视化接口，支持灵活配置数据源（train/test）、绘图方式（scatter/mesh）、输出内容（spotter/sniper）以及智能参考解对比。

## 核心原则
1. **向后兼容**: 不破坏现有功能，逐步迁移
2. **配置驱动**: 通过config.json控制可视化行为
3. **维度智能**: 根据解和参考解的维度自动选择最佳对比方式
4. **渐进实施**: 先在pseudotime完整实现，再推广到其他求解器

---

## 已完成 ✅

### 1. 核心数据结构设计
**文件**: `src/abstract_class/config/visualization_config.py`

- [x] `VisualizationConfig` 数据类
  - `dataset_mode`: "train_scatter" | "test_mesh" | "both"
  - `outputs`: ["spotter", "sniper", ...]
  - `reference_policy`: "auto" | "force" | "skip"
  - 动画、DPI、颜色等高级配置

- [x] `VisualizationRequest` 数据结构
  - 封装坐标、预测字典、参考解
  - 提供统一的数据传递接口

### 2. Pseudotime初步修复
- [x] 修改solver.py让未收敛也生成图片
- [x] 将animation数据从train改为test（line 244-251）
- [x] 更新visualizer参数名data_train→data

---

## 待实现任务

### Phase 1: 配置系统集成 (当前阶段)

#### 1.1 在config.json中添加visualization配置块
**优先级**: 🔴 高
**预计时间**: 30分钟

**任务内容**:
```json
{
  "visualization": {
    "dataset_mode": "test_mesh",
    "outputs": ["spotter", "sniper"],
    "reference_policy": "auto",
    "realtime_visualization": false,
    "save_animation": true,
    "animation_skip": 10
  }
}
```

**实施文件**:
- `cases/nonlinear_pde_cases/pseudotime_pde_cases/New_test_2D/config.json`
- `cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D/config.json` (后续)

**验证方法**:
- 加载配置不报错
- 打印配置值确认读取成功

---

#### 1.2 在PseudoTimePDEConfig中加载visualization配置
**优先级**: 🔴 高
**预计时间**: 45分钟

**任务内容**:
1. 在`_solver_specific_init()`中添加`_load_visualization_config()`
2. 实现加载逻辑（参考`_load_pseudo_time_config()`）
3. 提供默认值保证向后兼容

**实施文件**:
- `src/problem_solvers/nonlinear_pde_solver/pseudotime_pde_solver/utils/config.py`

**代码示例**:
```python
def _load_visualization_config(self):
    """加载可视化配置"""
    from src.abstract_class.config.visualization_config import VisualizationConfig

    # 从config中读取或使用默认值
    vis_dict = getattr(self, 'visualization', {})
    self.visualization = VisualizationConfig.from_dict(vis_dict)

    print(f"✓ Visualization config: mode={self.visualization.dataset_mode}, "
          f"outputs={self.visualization.outputs}")
```

**验证方法**:
- 运行求解器打印配置信息
- 测试缺少visualization块时使用默认值

---

### Phase 2: 智能参考解对比系统

#### 2.1 实现维度感知的参考解处理
**优先级**: 🟡 中高
**预计时间**: 2小时

**核心逻辑**:
```
数值解维度  参考解维度  处理策略
---------  ---------  ------------------
1D         1D         直接插值对比
1D         2D         参考解时间切片 → 1D对比
2D         2D         网格插值对比
2D         1D         数值解空间平均 → 1D对比 (较少见)
2D         3D         参考解时间切片 → 2D对比
```

**实施文件**:
- 新建 `src/abstract_class/utils/reference_solution_handler.py`

**关键函数**:
```python
class ReferenceSolutionHandler:
    @staticmethod
    def match_dimensions(U_numerical: np.ndarray,
                        x_numerical: np.ndarray,
                        ref_solution: Dict,
                        time_slice: float = None) -> Dict:
        """
        智能匹配数值解和参考解的维度

        Args:
            U_numerical: 数值解 (n_points, n_eqs)
            x_numerical: 数值解坐标 (n_points, n_dim)
            ref_solution: 参考解字典 {'x_ref', 'u_ref', 'y_ref'/'t_ref'?}
            time_slice: 时间切片位置（如果需要降维）

        Returns:
            处理后的参考解字典，维度与数值解匹配
        """
        n_dim_numerical = x_numerical.shape[1]
        has_time_dim = 'y_ref' in ref_solution or 't_ref' in ref_solution

        # 情况1: 都是1D
        if n_dim_numerical == 1 and not has_time_dim:
            return ref_solution

        # 情况2: 数值解1D，参考解2D → 时间切片
        if n_dim_numerical == 1 and has_time_dim:
            return ReferenceSolutionHandler._slice_2d_to_1d(
                ref_solution, time_slice
            )

        # 情况3: 都是2D
        if n_dim_numerical == 2 and has_time_dim:
            return ref_solution

        # 其他情况
        raise ValueError(f"Unsupported dimension combination")

    @staticmethod
    def _slice_2d_to_1d(ref_solution: Dict,
                       time_value: float) -> Dict:
        """从2D参考解中提取1D切片"""
        # 实现时间切片逻辑
        pass
```

**验证方法**:
- 单元测试：测试各种维度组合
- 集成测试：用实际case测试切片功能

---

#### 2.2 重构visualizer的参考解对比函数
**优先级**: 🟡 中高
**预计时间**: 1.5小时

**任务内容**:
1. 修改`plot_reference_comparison`使用`ReferenceSolutionHandler`
2. 根据`reference_policy`决定是否显示参考解
3. 没有参考解时只显示流场

**实施文件**:
- `src/problem_solvers/nonlinear_pde_solver/pseudotime_pde_solver/utils/visualize.py`

**修改点**:
```python
def plot_reference_comparison(self, U_steady, data, ref_solution, save_path):
    """智能参考解对比"""
    # 检查是否应该显示参考解
    if not self.config.visualization.should_show_reference(ref_solution is not None):
        # 只绘制流场
        self.plot_solution_field_only(U_steady, data, save_path)
        return

    # 使用ReferenceSolutionHandler匹配维度
    from src.abstract_class.utils.reference_solution_handler import ReferenceSolutionHandler

    matched_ref = ReferenceSolutionHandler.match_dimensions(
        U_steady, data['x'], ref_solution, time_slice=1.0
    )

    # 根据维度调用对应的绘图函数
    if self.n_dim == 1:
        self._plot_reference_comparison_1d(...)
    else:
        self._plot_reference_comparison_2d(...)
```

---

### Phase 3: 多源输出支持 (Spotter + Sniper)

#### 3.1 实现双输出绘图
**优先级**: 🟢 中
**预计时间**: 1小时

**任务内容**:
1. 根据`config.visualization.outputs`生成对应的图
2. Spotter输出：网络预测结果
3. Sniper输出：优化后结果

**文件命名规范**:
```
results/
  ├── spotter_solution.png          # 网络预测
  ├── spotter_vs_reference.png      # 网络vs参考解
  ├── sniper_solution.png           # Sniper优化后
  ├── sniper_vs_reference.png       # Sniper vs参考解
  └── convergence_history.png       # 收敛历史
```

**实施文件**:
- `src/problem_solvers/nonlinear_pde_solver/pseudotime_pde_solver/solver.py`

**代码逻辑**:
```python
def _postprocess_results(self, U_final, coeffs_final, conv_info):
    """后处理，生成所有配置的输出"""

    for output_name in self.config.visualization.outputs:
        if output_name == 'spotter':
            U_spotter = self._get_network_prediction()
            U_spotter_test = self._interpolate_to_test_grid(U_spotter)
            self.visualizer.plot_solution(
                U_spotter_test, self.data_test,
                f"results/spotter_solution.png"
            )

        elif output_name == 'sniper':
            U_sniper_test = self._interpolate_to_test_grid(U_final)
            self.visualizer.plot_solution(
                U_sniper_test, self.data_test,
                f"results/sniper_solution.png"
            )
```

---

#### 3.2 实现train_scatter和test_mesh的切换
**优先级**: 🟢 中
**预计时间**: 1小时

**任务内容**:
1. 根据`dataset_mode`决定使用哪个数据集
2. train用scatter，test用mesh
3. both模式生成两套图

**实施文件**:
- `src/problem_solvers/nonlinear_pde_solver/pseudotime_pde_solver/utils/visualize.py`

**修改函数**:
```python
def plot_solution(self, U, data, save_path):
    """根据配置选择绘图方式"""
    mode = self.config.visualization.dataset_mode

    if mode == "train_scatter":
        self._plot_scatter(U, data, save_path)
    elif mode == "test_mesh":
        self._plot_mesh(U, data, save_path)
    elif mode == "both":
        self._plot_scatter(U, data, save_path.replace('.png', '_scatter.png'))
        self._plot_mesh(U, data, save_path.replace('.png', '_mesh.png'))
```

---

### Phase 4: 扩展到其他求解器

#### 4.1 应用到FixIterationPDESolver
**优先级**: 🔵 低
**预计时间**: 1小时

**任务内容**:
- 复制pseudotime的配置加载逻辑
- 复用相同的visualizer改进

#### 4.2 应用到TimePDESolver
**优先级**: 🔵 低
**预计时间**: 1.5小时

**任务内容**:
- Time求解器有时间演化动画
- 需要特殊处理时空可视化

---

## 测试计划

### 单元测试
- [ ] `ReferenceSolutionHandler.match_dimensions()` 各种维度组合
- [ ] `VisualizationConfig.from_dict()` 配置加载
- [ ] 维度切片逻辑正确性

### 集成测试
- [ ] New_test_2D case完整运行
- [ ] 有参考解 vs 无参考解
- [ ] train_scatter模式
- [ ] test_mesh模式
- [ ] both模式

### 回归测试
- [ ] 老的case不加visualization配置仍能正常运行
- [ ] 所有求解器的现有测试通过

---

## 时间估算

| Phase | 任务 | 预计时间 |
|-------|------|---------|
| Phase 1 | 配置系统集成 | 1.5小时 |
| Phase 2 | 智能参考解对比 | 3.5小时 |
| Phase 3 | 多源输出支持 | 2小时 |
| Phase 4 | 扩展其他求解器 | 2.5小时 |
| 测试 | 单元+集成+回归 | 2小时 |
| **总计** | | **11.5小时** |

---

## 里程碑

1. **M1 - 基础配置** (完成Phase 1)
   - Config加载成功
   - 可以切换dataset_mode

2. **M2 - 智能对比** (完成Phase 2)
   - 维度自动匹配
   - 时间切片工作

3. **M3 - 完整功能** (完成Phase 3)
   - Spotter/Sniper双输出
   - Train/Test切换

4. **M4 - 全面推广** (完成Phase 4)
   - 所有求解器统一

---

## 后续优化 (可选)

1. **性能优化**
   - 大规模网格的绘图加速
   - 动画生成并行化

2. **功能扩展**
   - 交互式可视化(plotly)
   - 3D可视化支持
   - 误差场动画

3. **文档完善**
   - 配置说明文档
   - 可视化最佳实践
   - 示例gallery

---

## 注意事项

1. **向后兼容性**:
   - 所有改动都要保证老case不受影响
   - 提供默认值和降级路径

2. **代码质量**:
   - 添加类型注解
   - 编写docstring
   - 遵循项目代码规范

3. **测试覆盖**:
   - 关键函数必须有单元测试
   - 每个Phase完成后运行集成测试

4. **文档同步**:
   - 更新README说明新配置
   - 更新example案例
   - 记录breaking changes

---

## 参考资料

- 现有可视化代码:
  - `src/abstract_class/config/base_visualize.py`
  - `src/problem_solvers/nonlinear_pde_solver/pseudotime_pde_solver/utils/visualize.py`
  - `src/problem_solvers/nonlinear_pde_solver/fixiteration_pde_solver/utils/visualize.py`

- Codex分析报告: (本次对话记录)

- 相关Issue: (GitHub issue链接，如有)
