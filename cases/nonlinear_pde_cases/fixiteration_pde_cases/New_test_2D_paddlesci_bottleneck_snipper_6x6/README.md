# New_test_2D Paddle Bottleneck Snipper 6x6

这个算例把高精度 PaddleScience `bottleneck PirateNet` 直接接到 DeePoly `snipper`：

- `spotter` 使用已经训练好的 `256 -> 20` bottleneck PirateNet
- `snipper` 采用 `6 x 6` 时空分区
- 每个子区的局部基函数数量约为 `20 + (5 x 5) = 45`
  - `20` 个外部神经特征
  - `25` 个局部多项式特征，对应 `poly_degree = [4, 4]`
- `snipper_sampling.points_domain = 2400`
  - 相比旧的 `10000` 点主训练集已经显著压缩

建议命令：

```bash
python -m src.cli cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D_paddlesci_bottleneck_snipper_6x6
```

关键配置：

- `n_segments = [6, 6]`
- `poly_degree = [4, 4]`
- `spotter_feature_backend = "paddle"`
- `network_type = "bottleneck_piratenet"`
- `bottleneck_size = 20`
- `pretrained_model_path = ../../../../tmp/paddle_bottleneck_h256_b20_adam/checkpoints/best_model.pdparams`

当前默认只做 `1` 次 fixed-point / Snipper 重投影，便于单独观察高精度 spotter 进入分区最小二乘后的修正效果。
