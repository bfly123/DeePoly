# DeePoly 项目架构优化方案书

相关文档：

- 执行计划：[`docs/architecture-optimization-execution-plan.md`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/docs/architecture-optimization-execution-plan.md)
- 回归清单：[`docs/architecture-regression-checklist.md`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/docs/architecture-regression-checklist.md)

## 1. 文档目的

本文档基于项目当前架构分析结果与代码现状，给出一份可执行的架构优化方案书，用于指导 DeePoly 项目的分阶段重构、回归测试与风险控制。

文档目标：

- 明确当前项目的主要架构问题与风险来源
- 给出分阶段、低风险、可验证的优化路线
- 为每一部分改造定义严格的回归测试标准
- 保证优化过程不破坏现有 PDE 求解、可视化、自动代码生成与案例运行能力

## 1.1 最新执行状态（2026-03-19）

基于本轮持续优化后的实际代码状态，架构指标已明显高于初始分析结论：

- `architec` 仓库级指标：`8.86/10`，约等于 `88.6/100`
- `src` 核心代码指标：`9.0/10`，约等于 `90/100`
- 源码主干已经达到 `90` 分线，仓库级总分仍被 `cases/`、`examples/`、`docs/` 中的长文件和生成产物拖低

本轮已完成的关键结构优化包括：

- `BaseDataGenerator` 体系拆分为边界生成、边界投影、表达式求值、source/reference loader 等独立模块
- `BaseConfig` / `BasePDEConfig` 拆分出 JSON 读写、宽松加载、PDE 校验 mixin
- `auto_snipper` 拆分为导数提取和算子项分解两个 mixin
- `FixIterationPDEConfig` 拆分为 operator / loading / misc 三组 mixin
- 补充了 `visualization` dict/object 双格式兼容层，修复集成测试暴露的可视化兼容问题
- `TimePDEConfig` 拆分为 operator / scheme / misc 三组 mixin
- `auto_repalce_nonlinear.py` 拆分为方程代码生成与文件更新两个独立模块
- `TimePDEEvolutionMixin` 拆分为基础演化绘制和参考解演化绘制两个模块
- `auto_eq_parser.py` 拆分为线性项解析和非线性项解析两个模块

本轮新增或强化的回归覆盖包括：

- `test/test_config_base_helpers.py`
- `test/test_auto_snipper.py`
- `test/test_fixiteration_config.py`
- `test/test_base_data_helpers.py`
- `test/test_integration_with_solver.py`
- `test/test_time_config.py`
- `test/test_auto_replace_nonlinear.py`
- `test/test_time_pde_visualizer.py`
- `test/test_auto_eq.py`

## 2. 依据与输入

本方案主要基于以下材料：

- 架构分析结果： [`.architec/architec-analysis.json`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/.architec/architec-analysis.json)
- 仓库结构图： [`.hippocampus/structure-prompt.md`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/.hippocampus/structure-prompt.md)
- 关键入口与热点代码：
  - [`src/main_solver.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/main_solver.py)
  - [`src/__init__.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/__init__.py)
  - [`src/algebraic_solver/__init__.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/__init__.py)
  - [`src/algebraic_solver/fastnewton.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/fastnewton.py)
  - [`src/abstract_class/operator_usage_example.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/operator_usage_example.py)
  - [`src/abstract_class/config/base_data.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/config/base_data.py)
  - [`src/abstract_class/operator_factory.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/operator_factory.py)
  - [`src/meta_coding/auto_spotter.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/auto_spotter.py)
  - [`src/algebraic_solver/linear_solver.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/linear_solver.py)

## 3. 当前架构结论摘要

### 3.1 分析工具结论

根据架构分析与本轮实际复测：

- 初始 `architec-analysis.json` 结论：`65.76`
- 最新仓库级复测：`88.6/100`
- 最新 `src` 核心代码复测：`90/100`

说明：

- 当前项目目录拓扑整体可接受
- 问题主要不在“目录不合理”，而在“核心模块职责混杂、复杂度过高、依赖治理薄弱、测试边界不清”
- 当前剩余架构债务已经明显收敛到少数高复杂度可视化、时间积分和元编程模块

### 3.2 热点与高风险区域

当前剩余主要源码热点文件：

- [`src/problem_solvers/nonlinear_pde_solver/time_pde_solver/utils/time_visualize_animation_mixin.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/problem_solvers/nonlinear_pde_solver/time_pde_solver/utils/time_visualize_animation_mixin.py)
- [`src/problem_solvers/nonlinear_pde_solver/pseudotime_pde_solver/utils/pseudotime_visualize_animation_mixin.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/problem_solvers/nonlinear_pde_solver/pseudotime_pde_solver/utils/pseudotime_visualize_animation_mixin.py)
- [`src/meta_coding/nonlinear_equation_codegen_mixin.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/nonlinear_equation_codegen_mixin.py)
- [`src/meta_coding/auto_code_manager.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/auto_code_manager.py)

当前主要非源码拖分项：

- [`examples/operator_usage_example.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/examples/operator_usage_example.py)
- [`docs/regression-commands.md`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/docs/regression-commands.md)
- `cases/*/output.py`
- `cases/*/results/test_results_error.dat`

高风险组件：

- `src:abstract_class`
- `src:problem_solvers`
- `src:meta_coding`
- `src:algebraic_solver`

### 3.3 关键现象

当前架构的关键问题表现为：

- 核心抽象层混入示例代码、测试代码、验证代码
- 入口层、包初始化层存在硬耦合与副作用导入
- GPU 依赖泄漏到全局 import 链
- 元编程和算子构造大量依赖字符串处理与 `eval`
- 数据层承担过多职责
- 求解器与配置、后端、自动生成、可视化之间耦合偏重
- 测试主要集中在局部可视化，缺乏架构级回归门禁

## 4. 当前主要问题诊断

### 4.1 包初始化耦合过深

问题位置：

- [`src/__init__.py#L7`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/__init__.py#L7)
- [`src/algebraic_solver/__init__.py#L5`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/__init__.py#L5)
- [`src/algebraic_solver/fastnewton.py#L2`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/fastnewton.py#L2)

问题说明：

- `src` 顶层导入会递归导入 `algebraic_solver`
- `algebraic_solver` 初始化时会直接导入 `fastnewton`
- `fastnewton` 顶层直接依赖 `cupy`
- 结果是与 GPU 无关的测试或功能，只要 import 到 `src` 就可能失败

影响：

- CPU-only 环境无法稳定运行测试
- visualization 等模块被无关依赖拖垮
- 包边界不清晰，导入副作用过大

### 4.2 抽象层被示例/测试污染

问题位置：

- [`src/abstract_class/operator_usage_example.py#L1`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/operator_usage_example.py#L1)
- [`src/abstract_class/operator_usage_example.py#L248`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/operator_usage_example.py#L248)
- [`src/abstract_class/operator_usage_example.py#L2015`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/operator_usage_example.py#L2015)

问题说明：

- 该文件位于 `abstract_class`，理论上应承载稳定抽象
- 实际上它是“示例 + 校验 + 测试 + 教程 + 演示实现”的混合文件
- 文件超过 2000 行，复杂度和认知负担都过高

影响：

- 抽象层语义被破坏
- 复用与维护成本高
- 后续改动难以判断哪些是生产依赖，哪些只是示例验证

### 4.3 数据层职责过载

问题位置：

- [`src/abstract_class/config/base_data.py#L32`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/config/base_data.py#L32)
- [`src/abstract_class/config/base_data.py#L56`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/config/base_data.py#L56)
- [`src/abstract_class/config/base_data.py#L110`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/config/base_data.py#L110)
- [`src/abstract_class/config/base_data.py#L157`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/config/base_data.py#L157)

问题说明：

- 一个类同时负责：
  - case 自定义模块加载
  - 字符串数学表达式解析
  - piecewise 表达式处理
  - source term 读取与装配
  - reference solution 读取
- 同时存在 `sys.path` 注入和动态执行逻辑

影响：

- 职责过多，任何修改都容易引起连锁影响
- 表达式求值安全性差
- 单元测试难写，故障定位困难

### 4.4 算子构造器复杂度高且可诊断性差

问题位置：

- [`src/abstract_class/operator_factory.py#L25`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/operator_factory.py#L25)
- [`src/abstract_class/operator_factory.py#L102`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/operator_factory.py#L102)
- [`src/abstract_class/operator_factory.py#L77`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/operator_factory.py#L77)

问题说明：

- 非线性和线性算子都在单类中处理
- 求值前需要多轮扫描映射和字符串替换
- 存在裸 `except`，错误会被静默吞掉并退化为零值

影响：

- 算子错误难发现
- 性能不稳定
- 用户配置错误容易被隐藏

### 4.5 元编程层未形成中间表示

问题位置：

- [`src/meta_coding/auto_spotter.py#L38`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/auto_spotter.py#L38)
- [`src/meta_coding/auto_spotter.py#L75`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/auto_spotter.py#L75)
- [`src/meta_coding/auto_spotter.py#L128`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/auto_spotter.py#L128)
- [`src/meta_coding/auto_spotter.py#L212`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/auto_spotter.py#L212)

问题说明：

- 当前主要是字符串分析、字符串替换、代码字符串拼接
- 缺乏稳定的中间表示层，导致解析、转换、渲染混在一起
- 与 `auto_repalce_nonlinear.py` 存在潜在重复和分叉风险

影响：

- 难以扩展新方程格式
- 错误定位只能在生成结果层完成
- 生成代码的一致性难保障

### 4.6 线性求解器承担过多运行时职责

问题位置：

- [`src/algebraic_solver/linear_solver.py#L24`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/linear_solver.py#L24)
- [`src/algebraic_solver/linear_solver.py#L137`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/linear_solver.py#L137)
- [`src/algebraic_solver/linear_solver.py#L184`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/linear_solver.py#L184)

问题说明：

- 同时负责：
  - GPU 可用性探测
  - 设备选择
  - benchmark
  - 方法注册
  - fallback 策略
  - 线程池
  - 性能统计
  - 各种实际求解算法

影响：

- 类过大
- 后端演进困难
- 调试成本高

### 4.7 入口层混合环境准备、配置读取和业务调度

问题位置：

- [`src/main_solver.py#L17`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/main_solver.py#L17)
- [`src/main_solver.py#L50`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/main_solver.py#L50)
- [`src/main_solver.py#L118`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/main_solver.py#L118)

问题说明：

- `main_solver.py` 负责：
  - 修改 `sys.path`
  - 解析参数
  - 处理自动代码生成
  - 读取配置
  - 注册 solver
  - 创建结果目录
  - 调度 solver
- 同时存在硬编码默认 case

影响：

- CLI 不利于测试
- 分发和打包不稳定
- 入口层难以复用

## 5. 优化目标

### 5.1 架构目标

- 维持现有目录总体拓扑不大改
- 明确核心抽象层、运行时层、元编程层、后端层边界
- 消除包级副作用导入
- 将复杂热点拆解为可测试的协作对象
- 建立 CPU-only、GPU-optional 的稳定运行模式
- 将仓库级 `architec` 指标推进到 `90+`
- 将 `src` 核心实现指标稳定到 `90+`

### 5.2 工程目标

- 所有公共模块可以在无 `cupy` 环境下被安全导入
- 核心逻辑具备独立单元测试
- CLI、示例、自动代码生成、可视化具备最小可验证链路
- 形成分阶段回归测试门禁

### 5.3 非目标

本次优化不追求：

- 一次性重写全部 solver
- 一次性替换全部公式解析机制
- 大规模迁移目录或重命名全项目
- 在首轮重构中引入过重的框架依赖

## 6. 总体改造原则

### 6.1 原则一：先解耦，再优化

先消除耦合和副作用，再处理复杂度与性能。否则复杂度下降后仍可能因导入链或后端依赖失控而无法稳定运行。

### 6.2 原则二：先保护行为，再重构实现

每一轮重构前都先补最小行为测试，避免“边改边丢能力”。

### 6.3 原则三：生产代码与示例测试严格分层

- `src/` 仅放生产实现
- `test/` 放自动化测试
- `docs/examples/` 或 `examples/` 放教学示例

### 6.4 原则四：GPU 是可选能力，不是基础依赖

任何非 GPU 专属功能不应因缺失 `cupy` 而失效。

### 6.5 原则五：字符串处理进入边界层，核心层使用结构化对象

公式字符串、配置字符串、代码生成字符串都应尽量在边界层完成解析，核心层接收结构化 IR 或规范对象。

## 7. 分阶段实施方案

## 7.1 阶段一：导入链解耦与运行基线修复

### 7.1.1 目标

建立“无 GPU 也能 import、测试、运行非 GPU 功能”的基础工程环境。

### 7.1.2 修改范围

- [`src/__init__.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/__init__.py)
- [`src/algebraic_solver/__init__.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/__init__.py)
- [`src/algebraic_solver/fastnewton.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/fastnewton.py)
- 必要时补充轻量 API 封装文件

### 7.1.3 具体修改

1. 调整 `src/__init__.py`

- 不再 eager import 所有子包
- 只保留最轻量的包暴露能力
- 若需要，改为懒加载或显式子模块导入文档说明

2. 调整 `src/algebraic_solver/__init__.py`

- 不在包初始化时直接 import 所有 solver
- 将 `LinearSolver` 与 GPU 强依赖 solver 分离暴露

3. 调整 `fastnewton.py`

- 将 `cupy` 改为延迟导入
- 在无 `cupy` 环境下给出清晰错误信息或自动降级
- 不允许顶层 import 直接失败

4. 形成后端依赖约定

- `cpu-required` 路径必须无 GPU 依赖
- `gpu-optional` 路径必须通过 runtime 检测进入

### 7.1.4 严格回归测试

必须新增并通过以下测试：

1. 导入基线测试

- `python -c "import src"`
- `python -c "import src.visualization"`
- `python -c "from src.algebraic_solver.linear_solver import LinearSolver"`
- 无 `cupy` 环境下以上命令全部成功

2. pytest 收集测试

- `pytest --collect-only`
- 至少对 `test/test_dimensional_visualizer.py`、`test/test_substep_1_2.py`、`test/test_integration_with_solver.py` 收集成功

3. CPU-only 测试

- 在无 `cupy` 环境下运行可视化和基础工具测试
- 不允许在收集阶段因 GPU 缺失失败

4. 兼容性验证

- 若项目存在 GPU 环境，再补一轮 `cupy` 可用条件下的 smoke test

### 7.1.5 退出标准

- import 链不再因为 `cupy` 缺失而崩溃
- 所有与 GPU 无关测试可在 CPU-only 环境正常收集和执行

## 7.2 阶段二：抽象层净化与示例/测试剥离

### 7.2.1 目标

让 `abstract_class` 重新成为稳定的抽象层，而不是混合示例仓库。

### 7.2.2 修改范围

- [`src/abstract_class/operator_usage_example.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/operator_usage_example.py)
- `test/`
- `docs/` 或新增 `examples/`

### 7.2.3 具体修改

1. 文件拆分

- 生产相关逻辑保留在 `src/abstract_class/` 或新建更明确的运行时文件
- 示例迁移到 `docs/examples/` 或 `examples/`
- `test_*` 风格函数迁移到 `test/`

2. 边界明确

- `src/abstract_class/` 不允许保留演示型 print-heavy 测试代码
- 示例文档以 markdown 或独立脚本保留
- 测试必须可由 pytest 自动识别执行

3. 建立用途分类

- 生产实现
- 开发验证
- 教学示例

三者必须物理分离。

### 7.2.4 严格回归测试

1. 功能行为等价测试

- 从原文件提取出的关键示例场景必须变成可执行测试
- 至少覆盖：
  - simple multi-variable case
  - multi-equation case
  - precompiled operator case

2. 结果一致性测试

- 重构前后的 operator 输出形状一致
- 关键数值结果在容差内一致

3. 结构性测试

- `src/abstract_class/` 中不再出现 `test_*` 演示函数
- `pytest` 能在 `test/` 中发现对应替代测试

4. 文档示例验证

- 若保留示例脚本，则至少执行一次 smoke test

### 7.2.5 退出标准

- `abstract_class` 中只保留抽象与生产实现
- 原示例能力不丢失
- 原验证逻辑已转化为自动化测试或文档示例

## 7.3 阶段三：数据层职责拆解

### 7.3.1 目标

将 `BaseDataGenerator` 从大而全控制器改为编排器。

### 7.3.2 修改范围

- [`src/abstract_class/config/base_data.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/config/base_data.py)
- 各 solver 的 data generator 子类

### 7.3.3 建议拆分结构

建议引入以下协作对象：

- `CaseModuleLoader`
- `ExpressionEvaluator`
- `PiecewiseEvaluator`
- `SourceTermResolver`
- `ReferenceSolutionLoader`

`BaseDataGenerator` 只负责：

- 持有 config
- 调用上述对象
- 统一输出数据格式

### 7.3.4 具体修改

1. 去除直接 `sys.path` 注入

- 使用更明确的模块加载方式
- 兼容 case 目录下自定义脚本

2. 表达式求值隔离

- 将 `_parse_math_expression` 抽成独立 evaluator
- 明确支持的函数白名单
- 明确异常类型与错误信息

3. piecewise 逻辑隔离

- `_parse_piecewise_expression` 独立化
- 只负责 piecewise 的 mask 和组合，不混入其他职责

4. source term 解析隔离

- 支持字符串、数组、文件引用、自定义函数四类输入
- 每类解析逻辑分别测试

5. reference solution 读取隔离

- MAT 文件、自定义函数、配置表达式分别处理

### 7.3.5 严格回归测试

1. 表达式求值测试

- 常量表达式
- 一维表达式
- 二维表达式
- 多变量表达式
- 非法表达式报错测试

2. piecewise 测试

- 单区间
- 多区间
- 边界命中
- 未覆盖区间默认值

3. source term 测试

- `S` 为单字符串
- `S` 为列表
- `S` 为文件引用
- legacy `source_term`
- 无 source term 时返回零矩阵

4. 自定义 case 模块测试

- case 目录存在 `data_generate.py`
- case 目录不存在 `data_generate.py`
- 自定义模块缺少所需函数时应给出明确错误

5. 兼容性回归

- `func_fitting_solver`
- `linear_pde_solver`
- `time_pde_solver`

三个 data 子类必须通过最小数据生成 smoke test。

### 7.3.6 退出标准

- `BaseDataGenerator` 主体职责明显收敛
- 所有表达式与 source term 入口具备单元测试
- 子类无需感知内部重构细节

## 7.4 阶段四：算子构造器重构

### 7.4.1 目标

将当前“字符串替换 + 循环扫描 + 动态执行”的算子构建过程拆成可缓存、可测试、可诊断的结构。

### 7.4.2 修改范围

- [`src/abstract_class/operator_factory.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/abstract_class/operator_factory.py)

### 7.4.3 建议结构

建议拆为：

- `OperatorSpec`
- `OperatorExpressionCompiler`
- `OperatorVariableBinder`
- `OperatorExecutor`
- `OperatorFactory`

说明：

- `OperatorSpec`：结构化描述算子项
- `Compiler`：将符号表达式转为预处理表达式
- `Binder`：将 `features/u/coeffs` 绑定到变量
- `Executor`：执行表达式并返回标准化结果
- `Factory`：只做装配

### 7.4.4 具体修改

1. 预编译常量替换和数学函数映射

- 不在每次 operator 执行时重复正则替换

2. 建立变量索引表

- 避免每次扫描 `all_derivatives`
- 改成预先构建 derivative index 到变量映射

3. 禁止裸 `except`

- 任何表达式失败必须保留错误上下文
- 允许调用方决定是否降级

4. 统一输出协议

- 线性算子输出 shape 固定
- 非线性算子输出 shape 固定
- `F` 类算子输出单独明确定义

### 7.4.5 严格回归测试

1. 线性算子测试

- 单方程
- 多方程
- 多变量
- 不同 `var_idx`
- shape 校验

2. 非线性算子测试

- 只依赖零阶导
- 依赖 features 和 coeffs
- `segment_idx` 行为正确

3. 常量与数学函数测试

- `sin/cos/exp/log/sqrt`
- 常量替换与优先级

4. 错误诊断测试

- 非法表达式
- 缺失变量
- 维度不匹配

必须验证：

- 错误信息明确
- 不允许静默吞掉关键错误

5. 性能回归测试

- 对重构前后的典型 operator 调用进行 benchmark
- 要求不低于当前实现，或在可接受范围内

### 7.4.6 退出标准

- 算子工厂主类明显瘦身
- 关键场景有自动化测试
- 错误不再被静默吞掉

## 7.5 阶段五：元编程层重构

### 7.5.1 目标

将 `auto_spotter` 从字符串拼接脚本演进为“解析 -> 中间表示 -> 代码渲染”的稳定流水线。

### 7.5.2 修改范围

- [`src/meta_coding/auto_spotter.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/auto_spotter.py)
- [`src/meta_coding/auto_repalce_nonlinear.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/meta_coding/auto_repalce_nonlinear.py)

### 7.5.3 建议结构

建议拆分为：

- `EquationParser`
- `DerivativeUsageAnalyzer`
- `OperatorIRBuilder`
- `PytorchCodeRenderer`
- `CodeGenerationFacade`

### 7.5.4 具体修改

1. 统一 equation IR

- 表达式原文
- 变量集合
- 导数需求
- problem type
- operator bucket

2. 统一渲染出口

- 针对 time PDE、linear PDE、nonlinear PDE 共用一套渲染协议

3. 清理重复逻辑

- 与 `auto_repalce_nonlinear.py` 重复的解析逻辑收拢

4. 生成结果校验

- 生成前校验配置
- 生成后进行语法和关键符号验证

### 7.5.5 严格回归测试

1. equation parsing 测试

- 一阶导
- 二阶导
- 多变量
- 左右式带 `=`
- 零表达式

2. derivative analysis 测试

- 不同阶导被正确识别
- 高阶导自动包含低阶导依赖

3. code rendering 测试

- 生成的 PyTorch 导数代码包含预期变量
- 生成的 operator 代码结构正确

4. golden file 测试

- 选取典型 case 配置生成代码
- 与预期生成结果对比

5. autocode integration 测试

- 通过 `AutoCodeManager` 执行一次完整流程
- 确保生成结果可 import 或可执行

### 7.5.6 退出标准

- 元编程流程不再依赖混杂字符串流程推进
- 生成代码可测试、可比较、可追踪

## 7.6 阶段六：线性求解器后端化

### 7.6.1 目标

把 `LinearSolver` 从“大总管”重构成可扩展的后端选择系统。

### 7.6.2 修改范围

- [`src/algebraic_solver/linear_solver.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/algebraic_solver/linear_solver.py)

### 7.6.3 建议结构

- `LinearSolverConfig`
- `DeviceManager`
- `MethodSelector`
- `FallbackPolicy`
- `SolveStatsCollector`
- `DenseSolverBackend`
- `SparseSolverBackend`
- `GpuSolverBackend`

### 7.6.4 具体修改

1. 配置对象化

- 参数不直接散落在超长构造函数中

2. 设备管理独立

- GPU 可用性检查
- 显存策略
- batch 策略

3. 方法选择独立

- `auto/gpu_auto` 不在主 `solve()` 中铺开过多逻辑

4. fallback 策略独立

- 记录 fallback 原因
- 避免递归式黑盒 fallback

5. 统计与日志独立

- 性能信息收集不侵入主求解流程

### 7.6.5 严格回归测试

1. 求解正确性测试

- direct
- svd
- qr
- ridge
- lsqr
- lsmr

对同一输入矩阵进行解一致性比较。

2. fallback 测试

- 指定异常场景触发 fallback
- 验证 fallback 路径与日志

3. CPU-only 测试

- 无 `cupy` 环境下运行所有 CPU 方法

4. GPU-optional 测试

- 有 GPU 时验证 GPU 路径
- 无 GPU 时验证 graceful downgrade

5. 性能测试

- 大小不同矩阵下 benchmark
- 记录重构前后性能对比

### 7.6.6 退出标准

- `LinearSolver` 主类职责显著收敛
- CPU/GPU 路径行为清晰
- 方法选择与 fallback 可单测

## 7.7 阶段七：入口层与求解器注册体系整理

### 7.7.1 目标

把 CLI、配置读取、autocode、solver registry 彼此解耦。

### 7.7.2 修改范围

- [`src/main_solver.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/main_solver.py)
- [`src/problem_solvers/__init__.py`](/home/bfly/workspace/computeforcfd/混合网络/DeePoly_git/src/problem_solvers/__init__.py)

### 7.7.3 具体修改

1. 引入 `SolverRegistry`

- 由 registry 管理 problem_type 到 config/solver 的映射

2. `main_solver.py` 只做：

- 参数解析
- case 读取
- 调用应用服务层

3. autocode 独立为 pipeline stage

- 明确何时执行
- 明确是否 restart

4. 移除默认 case 硬编码

- 改为必填参数或明确默认策略

5. 逐步去除 `sys.path` 手工注入

### 7.7.4 严格回归测试

1. CLI smoke test

- 指定 func fitting case
- 指定 linear PDE case
- 指定 nonlinear PDE case

2. registry 测试

- 每个 problem type 能解析到对应 solver
- 错误类型返回清晰异常

3. autocode 测试

- 开启 autocode
- 不需要 autocode
- autocode 失败时行为

4. 结果目录测试

- 结果路径能创建
- 不覆盖已有关键结果

### 7.7.5 退出标准

- CLI 逻辑清晰
- solver 注册与扩展成本下降
- 入口层不再承担多重职责

## 8. 全局回归测试策略

## 8.1 回归测试分层

整个优化过程中，回归测试必须分为五层：

1. 导入层

- 检查包导入与依赖隔离

2. 单元层

- 检查小粒度对象行为

3. 组件层

- 检查 data/operator/codegen/solver backend 等组件组合行为

4. 集成层

- 检查从 case 配置到 solver 执行的短链路

5. 案例层

- 选取代表性 case 做端到端验证

## 8.2 建议建立的测试清单

### A. 导入与环境测试

- `import src`
- `import src.visualization`
- `import src.problem_solvers`
- `import src.meta_coding`
- CPU-only 环境下执行

### B. 单元测试

- expression evaluator
- piecewise evaluator
- operator compiler
- derivative analyzer
- method selector
- fallback policy

### C. 组件测试

- data generator
- operator factory
- auto code generation
- linear solver backend

### D. 集成测试

- func fitting 最小流程
- linear PDE 最小流程
- pseudotime/fixiteration 最小流程

### E. 端到端案例测试

建议至少固定 3 组 case：

- 一个函数拟合 case
- 一个线性 PDE case
- 一个非线性 PDE case

## 8.3 回归测试通过门禁

每个阶段合并前，必须满足：

- 新增测试全部通过
- 原有相关测试全部通过
- 无新增导入级失败
- 无新增核心 warning 泛滥
- 关键 case smoke test 通过

若涉及性能路径：

- 必须附 benchmark 对比

若涉及生成代码路径：

- 必须附 golden file 或产物一致性对比

## 9. 里程碑与建议排期

建议按以下顺序实施：

### M1：基线修复

- 导入链解耦
- GPU optional 化
- pytest 收集恢复

### M2：抽象层净化

- operator usage example 拆分
- 生产/示例/测试分层

### M3：核心热点拆解

- base_data 重构
- operator_factory 重构

### M4：元编程与后端整理

- auto_spotter IR 化
- linear_solver 后端化

### M5：入口层收口

- registry 化
- CLI 精简
- autocode pipeline 化

## 10. 风险与控制措施

### 10.1 风险：重构后行为偏移

控制措施：

- 重构前先补行为测试
- 对关键数值结果做容差断言

### 10.2 风险：示例迁移后能力丢失

控制措施：

- 所有示例必须变成文档脚本或自动化测试

### 10.3 风险：GPU 路径被削弱

控制措施：

- CPU-only 与 GPU-available 两类环境分别验证

### 10.4 风险：自动代码生成结果变化难以察觉

控制措施：

- 建立 golden file 与产物校验

### 10.5 风险：分阶段实施中间态不稳定

控制措施：

- 每阶段独立可合并
- 不在单个 PR 混入多个主题

## 11. 推荐交付物清单

实施本方案时，建议配套交付以下内容：

- 架构重构 PR 按阶段拆分
- 每阶段测试报告
- 关键 benchmark 对比报告
- 典型 case 回归记录
- 若有生成代码变化，提供生成产物对比记录

## 12. 最终建议

本项目当前最合理的优化路径不是“推倒重来”，而是“保留总体包结构，优先修复依赖治理和核心热点，再逐步结构化重构”。

优先级建议如下：

1. 先做导入链解耦与 GPU optional 化
2. 再净化 `abstract_class`
3. 然后处理 `base_data` 和 `operator_factory`
4. 再推进 `auto_spotter` 与 `linear_solver`
5. 最后收口 `main_solver` 与 registry

这样能以最小风险恢复工程稳定性，并为后续求解器能力演进建立清晰的基础架构。
