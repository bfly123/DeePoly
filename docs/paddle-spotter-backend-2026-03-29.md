# Paddle Spotter Backend 接入记录

日期：2026-03-29

## 本轮完成

已在 DeePoly 现有 external spotter provider 抽象上接入 Paddle 后端：

- `torch`
- `jax`
- `paddle`

对应代码：

- `src/abstract_class/spotter_feature_provider.py`
- `tools/paddlesci_feature_factory.py`
- `test/test_spotter_feature_provider.py`

## 当前实现方式

采用与 JAX backend 一致的工厂模式：

1. `config.json` 指定 `spotter_feature_backend = "paddle"`
2. `spotter_feature.factory_file` 指向外部 provider factory
3. factory 内部加载 PaddleScience 模型与预训练权重
4. 返回统一 provider，对外暴露：
   - `hidden_features(...)`
   - `hidden_feature_derivatives(...)`
   - `predict_output(...)`

这样 `snipper` / fitter 核心不需要改动。

## 已验证内容

### 1. Provider 单测

使用当前 conda Python 运行：

```bash
python -m pytest -q test/test_spotter_feature_provider.py
```

结果：

- `4 passed`
- `2 skipped`

说明：

- Torch provider 与旧路径结果一致
- Paddle provider 能计算 hidden feature、导数和输出
- 外部 factory 加载逻辑可用

### 2. PaddleScience Allen-Cahn 官方权重

在独立 Paddle 环境中已验证：

- Adam 预训练权重精确解误差约 `1.200156839332e-05`
- SOAP 预训练权重精确解误差约 `6.832714916527e-06`

同时确认该网络可以直接导出：

- `u`
- `u_t`
- `u_x`
- `u_xx`

这意味着它可以作为高精度 `spotter`，并不只是黑盒函数值提供器。

## 新增算例入口

新增 smoke case：

- `cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D_paddlesci_feature_smoke`

该算例使用：

- PaddleScience 官方 Allen-Cahn PirateNet SOAP 权重
- `allow_torch_training = false`
- DeePoly `run_snipper = true`

## 当前阻塞点

阻塞点不在 DeePoly 代码，而在运行环境：

- 当前 `base` 环境有 `torch`，但没有 `paddle` / `ppsci`
- 当前 `paddle_ac` 环境有 `paddle` / `ppsci`，但 `torch` 导入失败

因此当前状态是：

- 代码级集成完成
- 单元测试通过
- 官方 Paddle 模型可独立验证
- DeePoly 端到端 case 还缺一个“同时具备 Torch 与 PaddleScience 的运行环境”

## 推荐下一步

有两条路线：

### 路线 A：合并运行环境

建立一个统一环境，同时安装：

- `torch`
- `paddle`
- `ppsci`

优点：

- 实现最直接
- 可立即运行现有 Paddle smoke case

缺点：

- CUDA 依赖冲突风险较高

### 路线 B：做 Paddle 子进程桥接

保留当前主环境只负责 DeePoly + Torch，把 PaddleScience 放到独立子进程里，由 provider 通过 IPC / 文件缓存请求：

- hidden features
- derivatives
- output

优点：

- 彻底解耦 Torch 与 Paddle 依赖
- 更符合“并行 spotter”目标

缺点：

- 需要额外实现 RPC / 缓存 / 生命周期管理

如果后续目标是长期维护多个高精度 spotter 后端，路线 B 更稳。
