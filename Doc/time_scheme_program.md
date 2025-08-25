# IMEX-RK(2,2,2) 时间积分格式实现文档

## 数学公式

### 阶段 1 (Stage 1)
隐式求解系数 $\boldsymbol{\beta}^{(1)}$：

$$\left[\mathbf{V} - \gamma \Delta t \mathbf{L}_1 - \gamma \Delta t \mathbf{L}_2 \odot \mathbf{F}(\mathbf{U}^n)\right] \boldsymbol{\beta}^{(1)} = \mathbf{U}^n + \gamma\Delta t \mathbf{N}(\mathbf{U}^n)$$

计算阶段解：
$$\mathbf{U}^{(1)} = \mathbf{V} \boldsymbol{\beta}^{(1)}$$

### 阶段 2 (Stage 2)
隐式求解系数 $\boldsymbol{\beta}^{(2)}$：

$$\begin{aligned}
\left[\mathbf{V} - \gamma \Delta t \mathbf{L}_1 - \gamma \Delta t \mathbf{L}_2 \odot \mathbf{F}(\mathbf{U}^{(1)})\right] \boldsymbol{\beta}^{(2)} = \mathbf{U}^n &+ \Delta t(1-2\gamma) \left[\mathbf{L}_1 + \mathbf{L}_2 \odot \mathbf{F}(\mathbf{U}^{(1)}) \right] \boldsymbol{\beta}^{(1)}\\
&+ \Delta t(1-\gamma) \mathbf{N}(\mathbf{U}^{(1)})
\end{aligned}$$

计算阶段解：
$$\mathbf{U}^{(2)} = \mathbf{V} \boldsymbol{\beta}^{(2)}$$

## 变量维度说明

对于段索引 `segment_idx`：

| 变量 | 程序名称 | 维度 | 说明 |
|------|----------|------|------|
| $\mathbf{V}$ | `features[segment_idx]` | `[n_points, dgN]` | 特征矩阵（基函数） |
| $\mathbf{L}_1$ | `L1_operators[segment_idx]` | `[n_points, dgN]` | 线性算子1（如扩散项） |
| $\mathbf{L}_2$ | `L2_operators[segment_idx]` | `[n_points, dgN]` | 线性算子2（半隐式项） |
| $\mathbf{U}$ | `U_seg[segment_idx]` | `[n_points, n_eqs]` | 解变量 |
| $\mathbf{F}(\mathbf{U})$ | `F_func(features, U_seg)` | `[n_points, n_eqs]` | 非线性函数 |
| $\mathbf{N}(\mathbf{U})$ | `N_func(features, U_seg)` | `[n_points, n_eqs]` | 非线性算子 |
| $\boldsymbol{\beta}$ | `coeffs` | `[n_eqs, dgN]` | 展开系数 |

## 实现要点

### 1. 算子特性
- **L1, L2算子**：与特征矩阵 $\mathbf{V}$ 维度相同，对所有方程通用
- **F, N算子**：输出维度与解变量 $\mathbf{U}$ 相同
- **系数矩阵**：按方程分组，维度为 `[n_eqs, dgN]`

### 2. 特殊处理
由于 $(\mathbf{L}_2 \boldsymbol{\beta}) \odot \mathbf{F}$ 中无法直接提取 $\boldsymbol{\beta}$，需要：
- 对每个方程 `eq_idx` 单独处理
- 分别计算每个方程的系数和非线性项

### 3. 矩阵构建
雅可比矩阵的构建形式：
$$\mathbf{J} = \mathbf{V} - \gamma \Delta t \mathbf{L}_1 - \gamma \Delta t \mathbf{L}_2 \odot \text{diag}(\mathbf{F})$$

其中 $\text{diag}(\mathbf{F})$ 表示将 $\mathbf{F}$ 的值作为对角矩阵处理。
