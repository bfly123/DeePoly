# Multi-Backend Spotter Runtime

日期：2026-03-29

## 目标

在不破坏 `snipper` 主求解逻辑的前提下，将 `spotter` 升级为可并行编排的多后端运行时：

- `torch`
- `jax`
- `paddle`

核心要求：

- `snipper` 只消费统一特征接口
- 后端差异不进入 `fitter` 核心
- `jax` / `paddle` 可以在独立 Python 环境中运行
- 支持多候选并行评估与自动选优

## 当前实现

新增运行时包：

- `src/spotter_runtime/types.py`
- `src/spotter_runtime/selector.py`
- `src/spotter_runtime/orchestrator.py`
- `src/spotter_runtime/registry.py`
- `src/spotter_runtime/runtime/subprocess_runner.py`
- `src/spotter_runtime/workers/worker_main.py`

现有 solver 已接入该运行时：

- 初始化阶段可创建 `SpotterOrchestrator`
- `parallel_select` 模式下会创建多个候选 provider
- 候选按统一指标评分并选出当前最佳 provider
- 若后续发生 torch spotter 训练更新，会自动触发一次重选

## 配置格式

推荐新配置：

```json
{
  "spotter": {
    "mode": "parallel_select",
    "selector": {
      "metric": "test_l2",
      "max_points": 2048
    },
    "candidates": [
      {
        "name": "torch_local",
        "backend": "torch",
        "runtime": "inprocess",
        "priority": 0,
        "allow_torch_training": true
      },
      {
        "name": "jax_worker",
        "backend": "jax",
        "runtime": "subprocess",
        "priority": 1,
        "python": "/path/to/jax/env/bin/python",
        "feature": {
          "factory_file": "../../../../tools/jaxpi_feature_factory.py",
          "factory_function": "build_provider",
          "factory_kwargs": {}
        }
      },
      {
        "name": "paddle_worker",
        "backend": "paddle",
        "runtime": "subprocess",
        "priority": 2,
        "python": "/path/to/paddle/env/bin/python",
        "feature": {
          "factory_file": "../../../../tools/paddlesci_feature_factory.py",
          "factory_function": "build_provider",
          "factory_kwargs": {}
        }
      }
    ]
  }
}
```

兼容性说明：

- 若未提供 `spotter` 段，系统仍走旧的 `spotter_feature_backend + spotter_feature` 单后端路径
- 因此旧 case 不需要立即修改

## 选择逻辑

当前 selector 已支持：

- `test_l2`
- `reference_l2`
- `priority`
- `auto`

默认行为：

1. 优先使用 `data_test["x"] + data_test["u"]`
2. 若无测试真值，则尝试使用 `reference_solution`
3. 若仍无可比对数据，则按 `priority` 退化选择

## 子进程模式

`subprocess` runtime 的设计目标是隔离后端依赖，尤其是：

- Torch 主环境
- JAX 独立环境
- PaddleScience 独立环境

当前 worker 模式特点：

- 主进程通过 pickle IPC 调用 worker
- worker 内部可直接加载 `jax` 或 `paddle` provider
- `src.abstract_class` 和 `src.spotter_runtime` 已改为 lazy import，避免在非 torch 环境中包初始化时提前导入 `torch`

## 已完成验证

### 1. Spotter 单元测试

```bash
python -m pytest -q test/test_spotter_feature_provider.py
```

当前结果：

- `5 passed`
- `4 skipped`

### 2. Paddle 子进程跨环境 smoke

已验证以下路径：

- 主进程：当前默认 DeePoly 环境
- 子进程：`/home/bfly/anaconda3/envs/paddle_ac/bin/python`
- backend：`paddle`
- runtime：`subprocess`

结果表明：

- 主进程不需要安装 Paddle
- 子进程可独立返回 hidden features 与预测值

### 3. Torch + Paddle 并行选优 smoke

已验证：

- `torch_local` 与 `paddle_worker` 同时存在
- selector 基于 `test_l2` 自动选中更优的 `paddle_worker`

## 当前限制

- `parallel_select` 目前主要是“并行候选选择”，不是特征融合
- `selector` 目前只做误差型评分，还未纳入 PDE residual / boundary residual
- `jax` / `paddle` worker 目前通过通用 IPC 调用，功能完整，但还没有做大规模性能优化

## 下一步建议

如果后续进入大规模算例调优，建议继续做三件事：

1. 将 selector 指标扩展为：
   - PDE residual
   - boundary residual
   - derivative consistency
   - runtime cost

2. 为 `jaxpi` 与 `paddlesci` 增加官方 case 模板

3. 对 `subprocess` worker 增加批量缓存与请求合并，降低频繁导数查询的通信成本
