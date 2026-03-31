# Cases 目录分析与整理建议

## 1. 结论

`cases/` 当前不是单一职责目录，而是“输入案例库 + 参考数据仓 + 历史运行结果集”的混合体。

这会带来三个直接问题：

- 仓库级架构指标被大量非源码产物拖低
- 新案例难以判断哪些文件是运行必需，哪些是历史结果
- 后续自动化回归容易误把历史结果当成稳定输入

## 2. 当前结构

顶层分类如下：

- `cases/func_fitting_cases/`
- `cases/linear_pde_cases/`
- `cases/nonlinear_pde_cases/`

其中：

- `func_fitting_cases/` 共 6 个案例目录，其中 1 个为空目录
- `linear_pde_cases/` 共 4 个案例目录
- `nonlinear_pde_cases/` 当前主要包含：
  - `fixiteration_pde_cases/New_test_2D`
  - `pseudotime_pde_cases/New_test_2D`

## 3. 资产类型统计

当前 `cases/` 下主要文件类型：

- `.png`: 34
- `.npy`: 19
- `.py`: 18
- `.txt`: 14
- `.md`: 14
- `.json`: 13
- `.pt`: 9
- `.dat`: 9
- `.m`: 6
- `.mat`: 6
- `.npz`: 3
- `.gif`: 1

这说明 `cases/` 中已经包含大量运行结果与分析产物，而不仅是案例输入。

## 4. 结构模式分析

### 4.1 函数拟合案例

典型结构：

```text
case_name/
  config.json
  data_generate.py
  output.py
  results/
```

特点：

- `data_generate.py` 和 `output.py` 是运行链路的一部分
- `results/` 下同时存在模型、误差表、图像、数组数据

### 4.2 线性 PDE 案例

典型结构：

```text
case_name/
  config.json
  data_generate.py
  output.py
  results/
```

特点：

- 与函数拟合案例结构接近
- `results/README.md` 已在部分案例中存在，说明这些目录已经开始出现“结果说明”需求

### 4.3 非线性 PDE 案例

典型结构：

```text
case_name/
  config.json
  reference_data/
  results/
```

特点：

- `reference_data/` 中存放 `.mat`、`.m` 等静态参考资产
- `results/` 中存放 `.npz`、`.png`、`.gif` 等运行输出
- 这一类分层相对更清楚，整理成本最低

## 5. 当前主要问题

### 5.1 输入与结果混存

问题表现：

- `cases/` 同时保存可执行输入与历史运行结果
- 大量 `results/*.npy/*.npz/*.pt/*.png/*.dat` 已被纳入仓库视野

影响：

- repo 级架构分数被持续拉低
- 案例目录变得难以审查和维护

### 5.2 命名语义不一致

问题表现：

- `output.py` 实际是案例逻辑模块，而不是纯结果文件
- `results/` 中既有原始数值文件，也有报告、图片、模型

影响：

- 新贡献者难以判断什么应该提交、什么不该提交

### 5.3 空目录与孤立案例

问题表现：

- `cases/func_fitting_cases/case_2d_large_gradient/` 当前为空

影响：

- 会给维护者造成“案例缺失还是预留目录”的歧义

## 6. 运行依赖分析

当前代码对 case 目录有明确约束：

- 运行入口依赖 `case_dir/config.json`
- 函数拟合和线性 PDE solver 依赖 `case_dir/output.py`
- 函数拟合和线性 PDE 案例通常依赖 `case_dir/data_generate.py`
- 非线性 PDE 可视化和后处理依赖 `case_dir/reference_data/`

因此当前阶段不建议直接迁移已有案例目录路径，也不建议重命名 `output.py`。

## 7. 整理建议

### 7.1 立即执行

- 为 `cases/` 增加总览文档
- 为 `cases/` 增加结果产物忽略规则
- 明确 `results/` 只承载运行产物
- 明确 `reference_data/` 只承载静态参考输入

### 7.2 下一阶段

- 给每个案例大类补 `README.md`
- 清理不需要长期保留的历史结果产物
- 为 smoke 回归保留最小必要样例
- 对空目录给出说明或移除

### 7.3 不建议立即执行

- 不建议直接批量迁移 case 路径
- 不建议直接重命名 `output.py`
- 不建议在没有兼容层的前提下改变 solver 对案例目录的查找逻辑

## 8. 建议的治理目标

治理完成后，`cases/` 应当满足：

- 案例输入文件一眼可辨
- 参考资产与运行结果边界清晰
- 历史结果不会持续污染仓库结构质量
- smoke 测试只依赖最小必要案例
