# New_test_2D PaddleScience Smoke

这个算例用于验证 DeePoly 的 `spotter_feature_backend = "paddle"` 调用链。

目标：

- 使用 PaddleScience 官方 Allen-Cahn PirateNet 预训练权重作为 `spotter`
- 关闭 Torch 侧 spotter 训练
- 保留 DeePoly 当前 `snipper` / 分段拟合主流程

当前配置说明：

- `spotter_feature_backend = "paddle"`
- `spotter_feature.runtime = "subprocess"`
- `factory_file = "../../../../tools/paddlesci_feature_factory.py"`
- 预训练权重使用官方 SOAP 版本
- `coordinate_order = [1, 0]`
  - DeePoly 本算例坐标顺序是 `(x, y)`
  - PaddleScience Allen-Cahn 模型输入顺序是 `(t, x)`
  - 这里将 DeePoly 的 `(x, y)` 映射为 Paddle 的 `(x, t)` 物理顺序后，再按 `(t, x)` 送入网络

运行前提：

- 当前主环境需要具备 DeePoly 运行依赖与 `torch`
- `spotter_feature.python` 指向的子环境需要具备：
  - `paddle`
  - `ppsci`
- 因为 Paddle `spotter` 已经通过子进程桥接隔离，主环境和 Paddle 环境不必合并

说明：

- 这个 smoke case 主要验证 Paddle 预训练网络接入与 2x2 Snipper 主流程
- 如果要测试“Spotter 高精度预测 + 独立低采样 Snipper 重投影”，请直接使用同目录下的
  `New_test_2D_paddlesci_feature_resampled_snipper`

建议命令：

```bash
python -m src.cli cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D_paddlesci_feature_smoke
```
