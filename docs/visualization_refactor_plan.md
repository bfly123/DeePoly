# 可视化系统重构计划

## 目标架构

```
src/visualization/
├── core/
│   ├── __init__.py
│   ├── base.py                 # PlotComponent, FigureContext基类
│   ├── registry.py             # PlotRegistry组件注册表
│   ├── data_context.py         # DataContext数据源管理
│   └── figure_builder.py       # FigureBuilder布局构建器
├── components/
│   ├── __init__.py
│   ├── base_1d.py             # 1D组件基类
│   ├── base_2d.py             # 2D组件基类
│   ├── line_comparison.py     # 曲线对比图
│   ├── error_curve.py         # 误差曲线
│   ├── contour_field.py       # 等高线场
│   ├── scatter_field.py       # 散点场
│   ├── vector_field.py        # 矢量场
│   ├── residual_history.py    # 残差历史
│   ├── convergence_plot.py    # 收敛分析
│   ├── histogram.py           # 直方图
│   └── animation.py           # 动画序列
├── transforms/
│   ├── __init__.py
│   ├── registry.py            # Transform注册表
│   ├── grid.py                # 网格重建
│   ├── segment.py             # 分段处理
│   └── metrics.py             # 误差计算等
├── layouts/
│   ├── __init__.py
│   ├── grid_layout.py         # 网格布局
│   ├── subplot_layout.py      # 子图布局
│   └── colorbar_manager.py    # Colorbar管理
├── runtime/
│   ├── __init__.py
│   ├── base_runtime.py        # RuntimeVisualizer基类
│   ├── iteration_monitor.py   # 迭代监控
│   └── convergence_monitor.py # 收敛监控
├── case/
│   ├── __init__.py
│   ├── base_case.py           # CaseVisualizer基类
│   ├── default_case.py        # DefaultCaseVisualizer
│   └── factory.py             # Case visualizer加载器
├── config/
│   ├── __init__.py
│   ├── schema.py              # Pydantic schema定义
│   ├── loader.py              # YAML配置加载
│   └── validator.py           # 配置验证
└── legacy/
    ├── __init__.py
    └── adapters.py            # 向后兼容适配器

cases/
└── <case_name>/
    ├── config.json            # 保留原有配置
    ├── visualization.yaml     # 新增：可视化配置
    └── visualize_case.py      # 可选：自定义可视化
```

## 实施步骤

### Phase 1: 核心架构 (Day 1-2)
1. 创建core模块
   - PlotComponent抽象基类
   - DataContext数据源管理
   - PlotRegistry组件注册表
   - FigureBuilder布局构建器

2. 创建配置系统
   - Pydantic schema定义
   - YAML加载和验证
   - 与现有VisualizationConfig集成

### Phase 2: 基础组件库 (Day 2-3)
1. 1D组件
   - LineComparison（复用solution_plot_mixin逻辑）
   - ErrorCurve（复用error_analysis_mixin逻辑）

2. 2D组件
   - ContourField（复用grid_reconstruction_mixin）
   - ScatterField

3. 统计组件
   - ResidualHistory（复用convergence_mixin）
   - ConvergencePlot

### Phase 3: Transform系统 (Day 3)
1. 数据转换
   - stack_segments（处理分段数据）
   - reconstruct_grid（网格重建）
   - compute_error（误差计算）

2. Transform注册表
   - 内置transforms
   - 用户自定义callback

### Phase 4: 双层可视化 (Day 4-5)
1. RuntimeVisualizer
   - 从现有visualizer提取动态监控功能
   - 实时绘图、迭代更新
   - 保持在solver/utils/visualize.py

2. CaseVisualizer
   - 基类和接口定义
   - DefaultCaseVisualizer使用组件系统
   - 工厂加载机制

### Phase 5: 高级功能 (Day 5-6)
1. 动画系统
   - AnimationComponent
   - 序列帧管理
   - GIF/MP4导出

2. 多场耦合
   - 支持多变量可视化
   - 共享colorbar
   - 布局协调

3. 物理量计算
   - 可扩展callback系统
   - 内置常用物理量

### Phase 6: 迁移和测试 (Day 6-7)
1. 迁移现有solver visualizers
   - FixIterationPDEVisualizer
   - PseudoTimePDEVisualizer
   - LinearPDEVisualizer等

2. 创建示例cases
   - 简单case（纯YAML配置）
   - 中等case（YAML + callback）
   - 复杂case（自定义CaseVisualizer）

3. 测试
   - 单元测试（组件、transforms）
   - 集成测试（完整流程）
   - 回归测试（与旧行为对比）

### Phase 7: 文档和工具 (Day 7)
1. 文档
   - 架构说明
   - 组件API文档
   - 配置schema文档
   - 迁移指南

2. 工具
   - 脚手架生成器
   - 配置验证工具
   - 可视化预览工具

## 关键设计决策

### 1. 数据流
```
Solver训练完成
    ↓
RuntimeArtifacts (包含x, U, 历史等)
    ↓
├─→ RuntimeVisualizer (动态监控，已在训练中显示)
└─→ CaseVisualizer.render(artifacts)
        ↓
    加载visualization.yaml
        ↓
    FigureBuilder根据配置构建
        ↓
    各PlotComponent渲染
        ↓
    保存图表
```

### 2. 配置优先级
```
1. visualization.yaml (case特定配置)
2. config.json中的visualization字段 (向后兼容)
3. 代码中的默认值
```

### 3. 渐进式定制
```
Level 1 (80%): 纯YAML
  - 选择组件类型
  - 配置参数
  
Level 2 (15%): YAML + Callback
  - @register_transform装饰器
  - 小函数计算衍生量
  
Level 3 (5%): 自定义类
  - 继承PlotComponent
  - 完全控制
```

## 向后兼容策略

1. 保留旧的visualizer接口
2. 提供适配器自动转换
3. 逐步废弃警告
4. 迁移工具辅助

## 风险和缓解

1. **风险**：重构范围大，可能影响现有功能
   **缓解**：保持旧接口，并行开发，充分测试

2. **风险**：YAML配置复杂度高
   **缓解**：提供模板、验证工具、清晰文档

3. **风险**：性能下降（组件系统开销）
   **缓解**：缓存、延迟加载、性能测试

## 成功标准

- [ ] 所有现有test cases能用新系统运行
- [ ] 80%的cases可以用纯YAML配置
- [ ] 性能不低于现有实现
- [ ] 完整文档和示例
- [ ] 通过所有测试
