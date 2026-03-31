# New_test_2D Paddle Bottleneck Snipper 6x6 5k p55

这个算例是 `New_test_2D_paddlesci_bottleneck_snipper_6x6` 的增强版：

- 时空分区仍为 `6 x 6`
- 采样点提高到 `5000`
- 局部多项式提高到 `poly_degree = [5, 5]`
- 每区局部基函数约为 `20 + 36 = 56`
  - `20` 个 Paddle bottleneck 特征
  - `36` 个局部多项式特征

建议命令：

```bash
python -m src.cli cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D_paddlesci_bottleneck_snipper_6x6_5k_p55
```

主要目标：

- 检查增加局部 `snipper` 基函数后，是否能超过当前 `spotter`
- 检查采样点提高到约 `5000` 后，局部最小二乘是否更稳定
