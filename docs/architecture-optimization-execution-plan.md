# DeePoly 架构优化执行计划

## 1. 文档目的

本文档是 [`docs/architecture-optimization-plan.md`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/docs/architecture-optimization-plan.md) 的执行版，用于将总体方案拆分为可落地的 PR、任务包、验收标准和实施顺序。

执行原则：

- 每个 PR 只解决一个主题
- 每个 PR 都必须自带回归测试
- 先修复导入链和工程基线，再重构热点实现
- 尽量保证每个阶段完成后仓库仍处于可运行、可测试状态

## 1.1 当前执行进展（2026-03-19）

已完成并经回归验证的主题：

1. `BaseDataGenerator` 体系拆分：边界生成、边界投影、表达式求值、source/reference loader 全部独立
2. `BaseConfig` / `BasePDEConfig` 体系拆分：JSON 读写、宽松配置加载、PDE 校验逻辑模块化
3. `auto_snipper` 体系拆分：导数分析与算子项分解分离
4. `FixIterationPDEConfig` 体系拆分：operator/loading/misc 三段式
5. 可视化兼容修复：`visualization` 允许 `dict` 和对象两种配置输入
6. `TimePDEConfig` 体系拆分：operator/scheme/misc 三段式
7. `auto_repalce_nonlinear.py` 体系拆分：方程代码生成与文件替换逻辑分离
8. `TimePDEEvolutionMixin` 体系拆分：基础演化绘制与参考解对比拆分
9. `auto_eq_parser.py` 体系拆分：线性项与非线性项解析独立

当前分数状态：

- 仓库级：`88.6/100`
- `src` 核心实现：`90/100`

当前剩余优先级最高的任务包：

1. `PR-next-1` 拆分 [`src/problem_solvers/nonlinear_pde_solver/time_pde_solver/utils/time_visualize_animation_mixin.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/problem_solvers/nonlinear_pde_solver/time_pde_solver/utils/time_visualize_animation_mixin.py)
2. `PR-next-2` 拆分 [`src/problem_solvers/nonlinear_pde_solver/pseudotime_pde_solver/utils/pseudotime_visualize_animation_mixin.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/problem_solvers/nonlinear_pde_solver/pseudotime_pde_solver/utils/pseudotime_visualize_animation_mixin.py)
3. `PR-next-3` 收敛 [`src/meta_coding/nonlinear_equation_codegen_mixin.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/nonlinear_equation_codegen_mixin.py) 与 [`src/meta_coding/auto_code_manager.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/auto_code_manager.py)
4. `PR-next-4` 治理 `examples/`、`docs/` 和 `cases/` 中影响仓库级指标的长文件与生成产物

## 2. 实施顺序总览

建议顺序：

1. `PR-0` 建立回归测试基线
2. `PR-1` 导入链解耦与 GPU optional 化
3. `PR-2` 抽象层净化与示例/测试剥离
4. `PR-3` 数据层职责拆解
5. `PR-4` 算子构造器重构
6. `PR-5` 元编程层重构
7. `PR-6` 线性求解器后端化
8. `PR-7` 入口层与注册体系整理

## 3. PR 分解

## 3.1 PR-0 回归基线建设

### 目标

在大规模重构前，先建立“最小但严格”的架构回归门禁，避免后续改造没有护栏。

### 修改范围

- `test/`
- `docs/`
- 必要时新增测试辅助脚本

### 任务清单

- 新增 import smoke tests
- 新增 CPU-only 基线测试
- 梳理现有可运行测试并分类
- 固化 3 个代表性 case 作为回归样本
- 确定 benchmark 输入矩阵和 operator 测试样本

### 必须产出

- `test/test_import_smoke.py`
- `test/test_cpu_only_smoke.py`
- 代表性 case 清单文档
- 回归命令清单文档

### 验收标准

- 无 `cupy` 环境下至少能完成 import smoke tests
- pytest 可对基础测试做成功收集
- 后续 PR 均可复用这套回归基线

## 3.2 PR-1 导入链解耦与 GPU optional 化

### 目标

解决当前 `import src` 会因 `cupy` 缺失而失败的问题。

### 重点文件

- [`src/__init__.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/__init__.py)
- [`src/algebraic_solver/__init__.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/__init__.py)
- [`src/algebraic_solver/fastnewton.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/fastnewton.py)

### 任务清单

- 删除顶层重量级 eager import
- 将 GPU 依赖改为延迟导入
- CPU-only 场景提供明确降级路径
- 补充 import 行为测试

### 关键风险

- 旧代码可能依赖顶层 `__init__` 的副作用导入
- 延迟导入后可能暴露隐藏的循环依赖

### 验收标准

- `python -c "import src"` 成功
- 可视化相关测试在无 `cupy` 环境下能收集
- 不影响 `LinearSolver` 的 CPU 使用路径

## 3.3 PR-2 抽象层净化与示例/测试剥离

### 目标

将 `abstract_class` 从混合用途目录恢复为生产抽象层。

### 重点文件

- [`src/abstract_class/operator_usage_example.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/operator_usage_example.py)

### 任务清单

- 提取生产依赖逻辑
- 将 `test_*` 内容迁移到 `test/`
- 将教学示例迁移到 `docs/examples/` 或 `examples/`
- 清理 `abstract_class` 中与生产无关内容

### 关键风险

- 某些 solver 可能隐式依赖该文件中的验证逻辑
- 拆分后可能出现示例遗漏

### 验收标准

- `src/abstract_class/` 不再包含大段示例/测试逻辑
- 原有关键 operator 验证场景已转为自动化测试

## 3.4 PR-3 数据层职责拆解

### 目标

降低 `BaseDataGenerator` 的复杂度和耦合度。

### 重点文件

- [`src/abstract_class/config/base_data.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/config/base_data.py)
- 各 solver 的 data generator 子类

### 任务清单

- 提取 `CaseModuleLoader`
- 提取 `ExpressionEvaluator`
- 提取 `PiecewiseEvaluator`
- 提取 `SourceTermResolver`
- 提取 `ReferenceSolutionLoader`
- 保留 `BaseDataGenerator` 作为编排入口

### 关键风险

- case 兼容性可能受影响
- 旧配置格式和 legacy `source_term` 路径可能被破坏

### 验收标准

- 表达式求值、source term、reference solution 都有独立单测
- 三类 solver 的 data path 均通过 smoke test

## 3.5 PR-4 算子构造器重构

### 目标

把 `operator_factory` 变成结构清晰、错误可诊断的组件。

### 重点文件

- [`src/abstract_class/operator_factory.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/operator_factory.py)

### 任务清单

- 引入 `OperatorSpec`
- 引入编译、绑定、执行三个独立步骤
- 去掉裸 `except`
- 统一输出协议
- 为线性/非线性算子建立基准测试

### 关键风险

- shape 兼容性被破坏
- 预编译策略可能引入数值差异

### 验收标准

- 关键 operator 场景输出与旧实现一致
- 性能不劣化到不可接受范围
- 错误信息可定位到具体 operator term

## 3.6 PR-5 元编程层重构

### 目标

让自动代码生成过程从字符串脚本升级为稳定流水线。

### 重点文件

- [`src/meta_coding/auto_spotter.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/auto_spotter.py)
- [`src/meta_coding/auto_repalce_nonlinear.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/auto_repalce_nonlinear.py)

### 任务清单

- 建立 equation IR
- 统一导数分析
- 统一代码渲染接口
- 建立 golden file 测试
- 清理重复逻辑

### 关键风险

- 生成代码与旧版本存在细微差异
- 某些 case 对旧生成格式耦合较深

### 验收标准

- 典型 case 的生成代码通过 golden file 比较
- autocode 集成流程可跑通

## 3.7 PR-6 线性求解器后端化

### 目标

将 `LinearSolver` 拆解为设备管理、方法选择和求解实现的组合系统。

### 重点文件

- [`src/algebraic_solver/linear_solver.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/linear_solver.py)

### 任务清单

- 引入配置对象
- 提取 `DeviceManager`
- 提取 `MethodSelector`
- 提取 `FallbackPolicy`
- 提取统计与日志模块
- 建立 CPU/GPU 分层测试

### 关键风险

- fallback 行为变化
- 性能回退
- 稀疏矩阵路径兼容性问题

### 验收标准

- 各主方法解的一致性通过
- CPU-only 和 GPU-available 行为清晰
- fallback 可测试可追踪

## 3.8 PR-7 入口层与注册体系整理

### 目标

降低 `main_solver.py` 的系统职责，建立清晰的 solver 注册机制。

### 重点文件

- [`src/main_solver.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/main_solver.py)
- [`src/problem_solvers/__init__.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/problem_solvers/__init__.py)

### 任务清单

- 引入 `SolverRegistry`
- 从 `main_solver.py` 中抽出 solver 映射
- 处理 autocode pipeline 化
- 去掉硬编码默认 case
- 尽量消除 `sys.path` 注入

### 关键风险

- CLI 兼容性变化
- 旧案例调用脚本受影响

### 验收标准

- 代表性 case CLI smoke test 通过
- registry 具备清晰错误提示
- 入口层代码明显简化

## 4. 分工建议

若按多人并行实施，建议采用以下切分：

- 工程基线负责人：`PR-0`、`PR-1`
- 核心抽象负责人：`PR-2`、`PR-3`、`PR-4`
- 代码生成负责人：`PR-5`
- 求解后端负责人：`PR-6`
- 入口与集成负责人：`PR-7`

若单人实施，则必须严格按顺序推进，不建议跳步。

## 5. 每个 PR 的统一模板

每个 PR 必须包含：

- 改动目标
- 涉及文件范围
- 不兼容行为说明
- 新增或修改的测试
- 运行过的验证命令
- 已知风险与后续待办

## 6. 合并门禁

所有 PR 合并前必须满足：

- 对应阶段的严格回归测试通过
- 无新增导入级故障
- 无关键 case 冒烟失败
- 若涉及性能路径，附 benchmark 对比
- 若涉及生成代码路径，附生成产物对比

## 7. 推荐执行节奏

建议执行节奏：

- 第 1 周：`PR-0` + `PR-1`
- 第 2 周：`PR-2`
- 第 3 周：`PR-3`
- 第 4 周：`PR-4`
- 第 5 周：`PR-5`
- 第 6 周：`PR-6`
- 第 7 周：`PR-7`

如果团队资源有限，可将 `PR-3` 与 `PR-4` 合并为一个阶段，但不建议将 `PR-1` 与后续复杂重构混在同一个 PR 中。
