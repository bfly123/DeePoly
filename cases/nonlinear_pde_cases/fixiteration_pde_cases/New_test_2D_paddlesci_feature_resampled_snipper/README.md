# New_test_2D PaddleScience Resampled Snipper

这个算例展示新的两阶段数据流：

- `spotter` 使用 PaddleScience 预训练 PirateNet 直接给出场预测
- `snipper` 不再复用 `spotter` 训练点，而是通过 `snipper_sampling` 生成独立点云
- 当前配置只做 `1` 步固定点 / 最小二乘重投影，用于验证重采样机制

当前默认设置：

- `points_domain = 900`
  - 这是 DeePoly 主训练数据集规模
- `snipper_sampling.points_domain = 900`
  - 这是 Snipper 实际求解使用的独立点集规模
- `snipper_sampling.boundary_points = 120`
- `snipper_sampling.points_per_swap = 8`
- `fixiteration.max_iterations = 1`
- `linear_method = "direct"`

经验结论：

- 这种“独立点云 Snipper”比直接在原训练点上做全场 1 步重投影更稳定
- 但对于已经达到 `1e-5` 量级的 Paddle spotter，Snipper 仍然会引入明显误差
- 如果要把 Snipper 点数压到 `6000` 以内，同时又使用很多 segment 和高维隐藏特征，必须同步检查矩阵是否仍然超定

建议命令：

```bash
python -m src.cli cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D_paddlesci_feature_resampled_snipper
```
