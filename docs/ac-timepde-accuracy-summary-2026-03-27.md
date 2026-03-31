# Allen-Cahn 1D Time-PDE 精度汇总

## 统计口径

- 本文只汇总当前仓库内已经跑通并生成 `error_statistics_report.txt` 的 1D Allen-Cahn 时间推进算例。
- 所有误差均来自各 case 自身 `results/error_statistics_report.txt`。
- `points_domain` 表示训练点数，不是参考解网格数。
- `Global L2` / `Global L∞` 表示全时空误差。
- `TimeAverage L2` / `Max L2` 分别表示时间平均 L2 误差和各时刻中的最大 L2 误差。
- 对 `Allen_Cahn/AC_equation_*` 这组 onestep case，误差统计是基于训练网格导出的预测解。

## A. Onestep Predictor: 不同训练点数、不同 `dt`

| Case | Scheme | Points | Segments | `dt` | Time steps | Global L2 | TimeAverage L2 | Max L2 | Global L∞ | Physical time (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AC_equation_100_0.1` | `onestep_predictor` | 100 | 3 | 0.1 | 11 | 2.271687e-02 | 2.104173e-02 | 3.199721e-02 | 7.058655e-02 | 2.77 |
| `AC_equation_100_0.01` | `onestep_predictor` | 100 | 3 | 0.01 | 101 | 6.891622e-04 | 6.103482e-04 | 1.868757e-03 | 1.765384e-02 | 9.00 |
| `AC_equation_500_0.05` | `onestep_predictor` | 500 | 10 | 0.05 | 21 | 6.720012e-03 | 6.294634e-03 | 9.382784e-03 | 2.305522e-02 | 6.51 |
| `AC_equation_500_0.01` | `onestep_predictor` | 500 | 10 | 0.01 | 101 | 3.189989e-04 | 3.026821e-04 | 4.406851e-04 | 1.759470e-03 | 20.84 |
| `AC_equation_500_0.005_20seg` | `onestep_predictor` | 500 | 20 | 0.005 | 201 | 3.084215e-04 | 1.399946e-04 | 2.214110e-03 | 3.766814e-02 | 62.07 |

### 观察

- 在 `100` 点下，把 `dt` 从 `0.1` 降到 `0.01`，精度显著提升，`TimeAverage L2` 从 `2.10e-02` 降到 `6.10e-04`。
- 在 `500` 点下，将 `dt` 从 `0.05` 降到 `0.01` 后，误差显著下降；继续减小到 `0.005` 并同步将分区数从 `10` 提高到 `20` 后，`TimeAverage L2` 进一步降到 `1.40e-04`。
- 但 `500 / dt=0.005 / 20seg` 的提升主要体现在平均误差层面，`Max L2` 和 `Global L∞` 反而明显变大，说明后期陡梯度阶段的局部峰值误差仍然存在。
- 因此若更关注全程稳定性与峰值误差，当前 `500 / dt=0.01 / 10seg` 仍是更均衡的设置；若更关注时间平均精度，则 `500 / dt=0.005 / 20seg` 更优。

## B. IMEX-RK(2,2,2): 固定 `dt=0.04` 下的点数缩放

| Case | Points | Segments | `dt` | Time steps | Global L2 | TimeAverage L2 | Max L2 | Global L∞ | Physical time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AC_equation_imex_rk222_1d_50pts_3seg` | 50 | 3 | 0.04 | 25 | 4.040495e-02 | 3.626573e-02 | 8.033008e-02 | 8.131674e-01 | 16.95 |
| `AC_equation_imex_rk222_1d_100pts_3seg` | 100 | 3 | 0.04 | 25 | 3.535081e-02 | 3.281679e-02 | 4.932692e-02 | 1.314855e-01 | 21.46 |
| `AC_equation_imex_rk222_1d_200pts_3seg` | 200 | 3 | 0.04 | 25 | 3.535901e-02 | 3.282478e-02 | 4.932658e-02 | 1.318892e-01 | 24.35 |
| `AC_equation_imex_rk222_1d_400pts_3seg` | 400 | 3 | 0.04 | 25 | 3.535885e-02 | 3.282460e-02 | 4.932806e-02 | 1.308789e-01 | 23.27 |

### 观察

- 从 `50 -> 100` 点，误差有明显改善。
- 从 `100 -> 200 -> 400` 点，在当前 `dt=0.04` 和网络设置下，误差几乎不再下降。
- 这说明该组 IMEX 快速配置已经不再主要受空间采样点数限制，更可能受时间步长或 Spotter 精度限制。

## C. IMEX-RK(2,2,2): 固定 100 点下的分区数影响

| Case | Points | Segments | `dt` | Global L2 | TimeAverage L2 | Max L2 | Global L∞ | Physical time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AC_equation_imex_rk222_1d_100pts_3seg` | 100 | 3 | 0.04 | 3.535081e-02 | 3.281679e-02 | 4.932692e-02 | 1.314855e-01 | 21.46 |
| `AC_equation_imex_rk222_1d_100pts_6seg` | 100 | 6 | 0.04 | 3.536445e-02 | 3.283006e-02 | 4.932628e-02 | 1.314841e-01 | 17.95 |
| `AC_equation_imex_rk222_1d_100pts_12seg` | 100 | 12 | 0.04 | 3.532747e-02 | 3.279278e-02 | 4.932478e-02 | 1.314404e-01 | 49.02 |

### 观察

- 在 100 点、`dt=0.04` 下，`3/6/12` 分区的精度几乎相同。
- `12` 分区并没有带来可观精度收益，但耗时显著增加。

## D. IMEX-RK(2,2,2): 高精度小时间步配置

| Case | Points | Segments | `dt` | Time steps | Global L2 | TimeAverage L2 | Max L2 | Global L∞ | Physical time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AC_equation_imex_rk222_1d` | 200 | 4 | 0.0025 | 81 | 2.281397e-03 | 2.121352e-03 | 3.185518e-03 | 8.866591e-03 | 199.13 |
| `AC_equation_imex_rk222_1d_200pts_highacc` | 200 | 4 | 0.0025 | 81 | 2.281397e-03 | 2.121352e-03 | 3.185518e-03 | 8.866591e-03 | 208.45 |

### 观察

- 将 `dt` 从 `0.04` 大幅减小到 `0.0025` 后，200 点 IMEX 算例的 `Global L2` 从约 `3.54e-02` 降到 `2.28e-03`。
- 代价是物理计算时间从二十多秒上升到约两百秒。

## E. 当前可直接引用的推荐结果

| 目标 | 推荐 case | 主要理由 |
| --- | --- | --- |
| 快速粗略计算 | `AC_equation_100_0.1` | 运行很快，适合作为演示或烟测 |
| 兼顾效率与精度 | `AC_equation_500_0.01` | 当前全程误差最均衡，`Max L2` 最低 |
| 更高平均精度 | `AC_equation_500_0.005_20seg` | `TimeAverage L2` 更优，但峰值误差偏大 |
| IMEX 快速模式对比 | `AC_equation_imex_rk222_1d_100pts_3seg` | 配置简单，能代表 `dt=0.04` 一组结果 |
| IMEX 高精度模式 | `AC_equation_imex_rk222_1d_200pts_highacc` | 当前 IMEX 系列中的最佳精度结果 |

## 备注

- `AC_equation_500_0.01` 的当前数值结果是可信的，但旧仓库里同名 case 的文本误差报告存在明显口径错误，不建议继续直接引用旧版 `error_statistics_report.txt` 做论文表格。
- 若后续需要论文版表格，建议直接从各 case 当前 `results/error_statistics_report.txt` 和 `reference_data/train_data.mat` 重算生成，以避免旧存档文本混淆。
