# 可视化系统重构任务清单

**日期**: 2025-10-18
**目标**: 重构可视化系统，实现组件化、配置化的可视化架构

---

## 🚀 Step 1: 核心基础架构 (Expanded)
Progress: 3/4 substeps (75%)

- [x] 1.1: 创建目录结构和PlotComponent基类 ✅ (38/40)
- [x] 1.2: 实现DataContext和PlotRegistry ✅ (40/40)
- [x] 1.3: 实现FigureBuilder布局构建器 ✅ (37/40)
- [▶️] 1.4: 实现Pydantic配置系统 (~5k)

**Substep 1.1** (~5k): 创建目录结构和PlotComponent基类
├─ What: 创建visualization模块目录，实现核心抽象基类
├─ Input: docs/visualization_refactor_plan.md, visualize_mixins/
├─ Action: 创建目录树，实现PlotComponent抽象基类和FigureContext
├─ Output: src/visualization/core/base.py, __init__.py文件
├─ Validation: 可导入PlotComponent类，有render抽象方法
└─ Clear: Yes

**Substep 1.2** (~6k): 实现DataContext和PlotRegistry
├─ What: 数据源管理和组件注册表
├─ Input: core/base.py
├─ Action: 实现DataContext类管理数据源，PlotRegistry装饰器注册组件
├─ Output: core/data_context.py, core/registry.py
├─ Validation: 可注册和检索组件，DataContext可管理数据
└─ Clear: Yes

**Substep 1.3** (~4k): 实现FigureBuilder布局构建器
├─ What: 构建matplotlib figure布局
├─ Input: core/base.py, core/registry.py
├─ Action: 实现FigureBuilder，支持grid和subplot布局
├─ Output: core/figure_builder.py
├─ Validation: 可创建figure并添加组件
└─ Clear: Yes

**Substep 1.4** (~5k): 实现Pydantic配置系统
├─ What: 配置schema定义和YAML加载
├─ Input: 无
├─ Action: 定义Pydantic schema，实现YAML加载器和验证器
├─ Output: config/schema.py, config/loader.py, config/validator.py
├─ Validation: 可加载YAML配置并验证
└─ Clear: Yes

---

## 📋 剩余步骤

- [ ] Step 2: 基础组件库 (~25k tokens)
- [ ] Step 3: Transform和Layout系统 (~18k tokens)
- [ ] Step 4: 双层可视化系统 (~22k tokens)
- [ ] Step 5: 迁移现有Visualizer (~20k tokens)
- [ ] Step 6: 测试和文档 (~15k tokens)

## 📊 规划摘要
- 当前: Step 1.1
- 总步骤数: 6
- 预估tokens: 120k / 120k (100% 预算)
- 创建时间: 2025-10-18

---

## 📋 Phase 1: 核心架构 (Day 1-2)

### Task 1.1: 创建目录结构
- [ ] 创建 `src/visualization/core/`
- [ ] 创建 `src/visualization/components/`
- [ ] 创建 `src/visualization/transforms/`
- [ ] 创建 `src/visualization/layouts/`
- [ ] 创建 `src/visualization/runtime/`
- [ ] 创建 `src/visualization/case/`
- [ ] 创建 `src/visualization/config/`
- [ ] 创建 `src/visualization/legacy/`

### Task 1.2: 实现核心组件
- [ ] 实现 PlotComponent 抽象基类 (core/base.py)
- [ ] 实现 DataContext 数据源管理 (core/data_context.py)
- [ ] 实现 PlotRegistry 组件注册表 (core/registry.py)
- [ ] 实现 FigureBuilder 布局构建器 (core/figure_builder.py)

### Task 1.3: 实现配置系统
- [ ] 创建 Pydantic schema 定义 (config/schema.py)
- [ ] 实现 YAML 配置加载 (config/loader.py)
- [ ] 实现配置验证器 (config/validator.py)

---

## 📋 Phase 2: 基础组件库 (Day 2-3)

### Task 2.1: 实现 1D 组件
- [ ] LineComparison 组件 (components/line_comparison.py)
  - 复用 solution_plot_mixin 逻辑
- [ ] ErrorCurve 组件 (components/error_curve.py)
  - 复用 error_analysis_mixin 逻辑

### Task 2.2: 实现 2D 组件
- [ ] ContourField 组件 (components/contour_field.py)
  - 复用 grid_reconstruction_mixin 逻辑
- [ ] ScatterField 组件 (components/scatter_field.py)

### Task 2.3: 实现统计组件
- [ ] ResidualHistory 组件 (components/residual_history.py)
  - 复用 convergence_mixin 逻辑
- [ ] ConvergencePlot 组件 (components/convergence_plot.py)

---

## 📋 Phase 3: Transform 系统 (Day 3)

### Task 3.1: 实现数据转换
- [ ] stack_segments 分段数据处理 (transforms/segment.py)
- [ ] reconstruct_grid 网格重建 (transforms/grid.py)
- [ ] compute_error 误差计算 (transforms/metrics.py)

### Task 3.2: 实现 Transform 注册表
- [ ] Transform 注册表系统 (transforms/registry.py)
- [ ] 用户自定义 callback 支持

---

## 📋 Phase 4: 双层可视化 (Day 4-5)

### Task 4.1: RuntimeVisualizer
- [ ] 实现 RuntimeVisualizer 基类 (runtime/base_runtime.py)
- [ ] 实现 IterationMonitor 迭代监控 (runtime/iteration_monitor.py)
- [ ] 实现 ConvergenceMonitor 收敛监控 (runtime/convergence_monitor.py)

### Task 4.2: CaseVisualizer
- [ ] 实现 CaseVisualizer 基类 (case/base_case.py)
- [ ] 实现 DefaultCaseVisualizer (case/default_case.py)
- [ ] 实现工厂加载机制 (case/factory.py)

---

## 📋 Phase 5: 高级功能 (Day 5-6)

### Task 5.1: 动画系统
- [ ] AnimationComponent 实现 (components/animation.py)
- [ ] 序列帧管理
- [ ] GIF/MP4 导出支持

### Task 5.2: 多场耦合
- [ ] 多变量可视化支持
- [ ] 共享 colorbar 管理 (layouts/colorbar_manager.py)
- [ ] 布局协调器

---

## 📋 Phase 6: 迁移和测试 (Day 6-7)

### Task 6.1: 迁移现有 visualizer
- [ ] 迁移 FixIterationPDEVisualizer
- [ ] 迁移 PseudoTimePDEVisualizer
- [ ] 迁移 LinearPDEVisualizer
- [ ] 迁移 FuncFittingVisualizer

### Task 6.2: 测试验证
- [ ] 创建简单测试案例（纯 YAML 配置）
- [ ] 创建中等测试案例（YAML + callback）
- [ ] 创建复杂测试案例（自定义 CaseVisualizer）
- [ ] 运行回归测试确保兼容性

---

## 当前状态

**进度**: 0/25 任务完成

**当前阶段**: 准备开始 Phase 1

**已完成工作**:
- [x] Mixin 模块化拆分到 `src/abstract_class/config/visualize_mixins/`
  - animation_mixin.py
  - convergence_mixin.py
  - error_analysis_mixin.py
  - grid_reconstruction_mixin.py
  - realtime_visualization_mixin.py
  - reference_comparison_mixin.py
  - segment_overlay_mixin.py
  - solution_plot_mixin.py
- [x] 保留 base_visualize_old_backup.py 作为备份

---

## 执行原则

1. **代码简洁**: 最简化代码设计
2. **直面问题**: 拒绝防御性编程，发现错误而不是加 if 分支
3. **组件复用**: 充分复用现有 mixin 中的逻辑
4. **逐步验证**: 每个 Task 完成后验证功能
