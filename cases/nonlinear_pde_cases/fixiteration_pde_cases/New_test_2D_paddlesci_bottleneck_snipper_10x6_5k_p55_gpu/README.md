# New_test_2D Paddle Bottleneck Snipper 10x6 5k p55 GPU

这个算例继续增加局部分区，重点提高空间方向分辨率：

- `n_segments = [10, 6]`
  - `x` 方向 `10` 段
  - `y` 方向 `6` 段
- `points_domain = 5000`
- `poly_degree = [5, 5]`
- 每区局部基函数约为 `20 + 36 = 56`
- 线性最小二乘强制使用 GPU：
  - `linear_device = "cuda"`
  - `linear_method = "gpu_direct"`

建议在 `ML` 环境中运行：

```bash
conda activate ML
python -m src.cli cases/nonlinear_pde_cases/fixiteration_pde_cases/New_test_2D_paddlesci_bottleneck_snipper_10x6_5k_p55_gpu
```
