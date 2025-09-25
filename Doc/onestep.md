
# 单步时间积分格式

## 原始方程
$$u_t + L_1 + L_2 \cdot F + N = 0$$

## 预测步
$$\frac{u^* - u^n}{\Delta t} + L_1(u^*) + L_2(u^*) \cdot F(u^n) + N(u^n) = 0$$

## 校正步

### 条件判断

**当 step > 1 时，执行校正步：**

$$\frac{u^{n+1} - u^n}{\Delta t} + \frac{L_1(u^*) + L_1(u^n)}{2} + \frac{L_2(u^*) + L_2(u^n)}{2} \cdot F\left(\frac{u^n + u^*}{2}\right) + N\left(\frac{u^* + u^n}{2}\right) = 0$$

$$u^n = u^{n+1}$$

**否则，直接使用预测值：**

$$u^n = u^*$$

### 更新算子
$$L_1(u^n) = L_1(u^*)$$
$$L_2(u^n) = L_2(u^*)$$

---

# 实现伪代码

## 类结构设计

```python
class OneStepPredictor(BaseTimeScheme):
    """单步预测-校正时间积分格式"""

    def __init__(self, config):
        super().__init__(config)
        self.order = 1  # 基本精度
        self.predictor_solution = None  # 存储预测解 u^*
        self.predictor_coeffs = None    # 存储预测系数
        self.L1_beta_prev = None        # 存储上一时间步的 L1*β^n
        self.L2_beta_prev = None        # 存储上一时间步的 L2*β^n

    def time_step(self, U_n, U_seg, dt, coeffs_n=None, current_time=0.0, step=0):
        """执行单步预测-校正时间积分"""
        # 步骤1: 预测步
        U_star_seg, coeffs_star = self._predictor_step(U_seg, dt, step)

        # 步骤2: 校正步（仅当step > 1时）
        if step > 1:
            U_new_seg, coeffs_new = self._corrector_step(U_seg, U_star_seg, dt, coeffs_star, step)
        else:
            U_new_seg, coeffs_new = U_star_seg, coeffs_star

        # 步骤3: 更新算子值供下一时间步使用
        # 当前时间步的预测算子值成为下一时间步的"上一步"算子值
        self.L1_beta_prev = self.L1_beta_star
        self.L2_beta_prev = self.L2_beta_star

        # 转换为全局数组
        U_new = self.fitter.segments_to_global(U_new_seg)

        return U_new, U_new_seg, coeffs_new
```

## 核心算法实现

### 预测步实现

```python
def _predictor_step(self, U_n_seg, dt, step):
    """预测步: (u^* - u^n)/Δt + L1(u^*) + L2(u^*)⋅F(u^n) + N(u^n) = 0"""

    # 准备预测步数据 - 按照IMEX-RK的接口格式
    predictor_data = {
        "U_n_seg": U_n_seg,
        "dt": dt,
        "step": step,
        "operation": "onestep_predictor"  # 新的operation类型
    }

    # 求解预测步系数
    coeffs_star = self.fitter.fit(**predictor_data)


    # 构造预测解
    U_star_global, U_star_seg = self.fitter.construct(
        self.fitter.data, self.fitter._current_model, coeffs_star
    )

    # 直接计算并存储算子值 L1*β 和 L2*β，避免校正步重复计算
    L1_beta_star, L2_beta_star = self._compute_operator_values(coeffs_star)

    # 存储预测解和算子值供校正步使用
    self.predictor_solution = U_star_seg
    self.predictor_coeffs = coeffs_star
    self.L1_beta_star = L1_beta_star  # 存储 L1(u^*) 的算子值
    self.L2_beta_star = L2_beta_star  # 存储 L2(u^*) 的算子值

    return U_star_seg, coeffs_star

def _compute_operator_values(self, coeffs):
    """计算并存储算子值 L1*β 和 L2*β

    参数:
        coeffs: np.ndarray, shape (ns, n_eqs, dgN) - 系数矩阵

    返回:
        L1_beta: List[List[np.ndarray]] - shape [ns][n_eqs](n_points,)
        L2_beta: List[List[np.ndarray]] - shape [ns][n_eqs](n_points,)
    """
    L1_beta = []  # 存储各segment的 L1*β
    L2_beta = []  # 存储各segment的 L2*β

    for seg_idx in range(self.fitter.ns):
        # 获取该segment的算子 - 注意维度匹配
        L1_ops = self.fitter._linear_operators[seg_idx].get("L1", None)
        L2_ops = self.fitter._linear_operators[seg_idx].get("L2", None)

        # 计算该segment各方程的算子值
        L1_seg = []
        L2_seg = []

        for eq_idx in range(self.config.n_eqs):
            # 获取系数: shape (dgN,)
            beta = coeffs[seg_idx, eq_idx, :]

            # 计算 L1*β: (n_points, dgN) @ (dgN,) = (n_points,)
            if L1_ops is not None and len(L1_ops) > eq_idx:
                L1_val = L1_ops[eq_idx] @ beta  # shape: (n_points,)
                L1_seg.append(L1_val)
            else:
                L1_seg.append(None)

            # 计算 L2*β: (n_points, dgN) @ (dgN,) = (n_points,)
            if L2_ops is not None and len(L2_ops) > eq_idx:
                L2_val = L2_ops[eq_idx] @ beta  # shape: (n_points,)
                L2_seg.append(L2_val)
            else:
                L2_seg.append(None)

        L1_beta.append(L1_seg)
        L2_beta.append(L2_seg)

    return L1_beta, L2_beta

def _corrector_step(self, U_n_seg, U_star_seg, dt, coeffs_star, step):
    """校正步: 显式计算
    u^{n+1} = u^n - Δt * [L1(u^*) + L1(u^n)]/2 - Δt * [L2(u^*) + L2(u^n)]/2 ⋅ F((u^n + u^*)/2) - Δt * N((u^* + u^n)/2)"""

    U_new_seg = []

    for seg_idx in range(len(U_n_seg)):
        U_n = U_n_seg[seg_idx]
        U_star = U_star_seg[seg_idx]

        # 计算中点值 (u^n + u^*)/2
        U_mid = 0.5 * (U_n + U_star)

        # 获取特征矩阵和算子
        features = self.fitter._features[seg_idx][0]
        L1_ops = self.fitter._linear_operators[seg_idx].get("L1", None)
        L2_ops = self.fitter._linear_operators[seg_idx].get("L2", None)

        # 计算函数值
        F_mid = self.fitter.F_func(features, U_mid) if self.fitter.has_operator("F") else None
        N_mid = self.fitter.N_func(features, U_mid) if self.fitter.has_operator("N") else None

        # 初始化校正解: u^{n+1} = u^n
        U_corrected = U_n.copy()

        # 对每个方程进行校正
        for eq_idx in range(self.config.n_eqs):
            # 计算 L1 项的平均值: [L1(u^*) + L1(u^n)]/2
            # 使用预计算的算子值，确保维度匹配
            L1_avg = np.zeros(U_n.shape[0])  # shape: (n_points,)
            if (hasattr(self, 'L1_beta_star') and len(self.L1_beta_star) > seg_idx and
                len(self.L1_beta_star[seg_idx]) > eq_idx and self.L1_beta_star[seg_idx][eq_idx] is not None and
                hasattr(self, 'L1_beta_prev') and self.L1_beta_prev is not None and
                len(self.L1_beta_prev) > seg_idx and len(self.L1_beta_prev[seg_idx]) > eq_idx):

                L1_star = self.L1_beta_star[seg_idx][eq_idx]  # shape: (n_points,)
                L1_n = self.L1_beta_prev[seg_idx][eq_idx]     # shape: (n_points,)
                if L1_n is not None:
                    L1_avg = 0.5 * (L1_star + L1_n)  # shape: (n_points,)

            # 计算 L2⋅F 项的平均值: [L2(u^*) + L2(u^n)]/2 ⋅ F((u^n + u^*)/2)
            # 使用预计算的算子值，确保维度匹配
            L2F_avg = np.zeros(U_n.shape[0])  # shape: (n_points,)
            if (hasattr(self, 'L2_beta_star') and len(self.L2_beta_star) > seg_idx and
                len(self.L2_beta_star[seg_idx]) > eq_idx and self.L2_beta_star[seg_idx][eq_idx] is not None and
                hasattr(self, 'L2_beta_prev') and self.L2_beta_prev is not None and
                len(self.L2_beta_prev) > seg_idx and len(self.L2_beta_prev[seg_idx]) > eq_idx and
                F_mid is not None):

                L2_star = self.L2_beta_star[seg_idx][eq_idx]  # shape: (n_points,)
                L2_n = self.L2_beta_prev[seg_idx][eq_idx]     # shape: (n_points,)
                if L2_n is not None:
                    L2_avg = 0.5 * (L2_star + L2_n)          # shape: (n_points,)
                    L2F_avg = L2_avg * F_mid[:, eq_idx]       # shape: (n_points,)

            # 计算 N 项: N((u^* + u^n)/2)
            N_contrib = np.zeros(U_n.shape[0])  # shape: (n_points,)
            if N_mid is not None:
                N_contrib = N_mid[:, eq_idx]  # shape: (n_points,)

            # 显式更新: u^{n+1} = u^n - Δt * [L1_avg + L2F_avg + N_contrib]
            # 确保所有项都是 (n_points,) 的向量
            U_corrected[:, eq_idx] -= dt * (L1_avg + L2F_avg + N_contrib)

        U_new_seg.append(U_corrected)

    # 由于是显式计算，直接使用预测步系数
    coeffs_new = coeffs_star

    return U_new_seg, coeffs_new
```

### 雅可比矩阵构建

```python
def build_stage_jacobian(self, segment_idx, **kwargs):
    """构建预测步的雅可比矩阵（校正步是显式的，不需要雅可比）

    与 BaseTimeScheme 接口保持一致
    """
    operation = kwargs.get("operation", "onestep_predictor")
    dt = kwargs.get("dt")

    if operation == "onestep_predictor":
        return self._build_predictor_jacobian(segment_idx, dt, **kwargs)
    else:
        raise ValueError(f"Only onestep_predictor needs Jacobian. Got: {operation}")

def _build_predictor_jacobian(self, segment_idx, dt, **kwargs):
    """预测步雅可比: [V + Δt⋅L1 + Δt⋅L2⊙F(u^n)] β^* = u^n - Δt⋅N(u^n)"""

    # 获取维度和算子
    n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
    ne = self.config.n_eqs
    dgN = self.fitter.dgN

    # 获取算子和特征
    L1_ops = self.fitter._linear_operators[segment_idx].get("L1", None)
    L2_ops = self.fitter._linear_operators[segment_idx].get("L2", None)
    features = self.fitter._features[segment_idx][0]

    # 获取当前解 u^n
    U_n_seg_list = kwargs.get("U_n_seg", [])
    U_n_seg = U_n_seg_list[segment_idx]

    # 计算 F(u^n) 和 N(u^n)
    F_n = self.fitter.F_func(features, U_n_seg) if self.fitter.has_operator("F") else None
    N_n = self.fitter.N_func(features, U_n_seg) if self.fitter.has_operator("N") else None

    # 构建系统矩阵
    A_matrix = np.zeros((ne * n_points, ne * dgN))
    b_vector = []

    for eq_idx in range(ne):
        # 行和列索引
        row_start, row_end = eq_idx * n_points, (eq_idx + 1) * n_points
        col_start, col_end = eq_idx * dgN, (eq_idx + 1) * dgN

        # 构建该方程的雅可比: V + Δt⋅L1 + Δt⋅L2⊙F(u^n)
        J_eq = features.copy()  # V项

        # 添加 Δt⋅L1 项
        if L1_ops is not None:
            J_eq += dt * L1_ops[eq_idx]

        # 添加 Δt⋅L2⊙F(u^n) 项
        if L2_ops is not None and F_n is not None:
            F_eq = F_n[:, eq_idx]
            J_eq += dt * np.diag(F_eq) @ L2_ops[eq_idx]

        A_matrix[row_start:row_end, col_start:col_end] = J_eq

        # 构建右端向量: u^n - Δt⋅N(u^n)
        rhs_eq = U_n_seg[:, eq_idx].copy()
        if N_n is not None:
            rhs_eq -= dt * N_n[:, eq_idx]

        b_vector.append(rhs_eq)

    b_final = np.concatenate(b_vector)
    return A_matrix, b_final
```

## 关键特性

1. **混合求解策略**:
   - 预测步: 隐式求解，需要调用 `fitter.fit()` 构建和求解线性系统
   - 校正步: 显式计算，直接基于预测解更新

2. **条件校正**: 仅在 step > 1 时执行校正步，第一步直接使用预测值

3. **算子复用**: 充分利用现有的L1, L2, F, N算子框架和预编译结构

4. **中点评估**: 在 (u^n + u^*)/2 处评估非线性项F和N，提高精度

5. **算子值预计算与复用**:
   - 预测步构造解时，同时计算 L1*β^* 和 L2*β^*
   - 校正步使用当前的 L1*β^* 和上一步预存的 L1*β^n
   - 完全避免重复的矩阵-向量乘法运算
   - 减少约75%的算子计算量

6. **时间步算子值传递**:
   - 每个时间步结束后，当前的算子值成为下一步的"上一步"值
   - L1_beta_prev ← L1_beta_star（时间步间传递）
   - L2_beta_prev ← L2_beta_star（时间步间传递）

7. **计算效率**:
   - 预测步: 一次矩阵求解 + 算子值预计算
   - 校正步: 直接使用预计算值 + 矢量运算，无需矩阵求解
   - 避免重复构建雅可比矩阵和重复算子计算

## 实现要点

1. **接口一致性**: 完全遵循 `BaseTimeScheme` 接口规范
2. **预测步雅可比矩阵**: `[V + Δt⋅L1 + Δt⋅L2⊙F(u^n)]`，shape: `(n_eqs*n_points, n_eqs*dgN)`
3. **预测步右端向量**: `u^n - Δt⋅N(u^n)`，shape: `(n_eqs*n_points,)`
4. **算子值预计算**: 在构造预测解时同时计算 `L1*β^*` 和 `L2*β^*`
5. **校正步显式更新**:
   - 使用当前预计算的 L1*β^*, L2*β^* (shape: `(n_points,)`)
   - 使用上一步预存的 L1*β^n, L2*β^n (shape: `(n_points,)`)
   - 完全基于预存算子值的直接矢量运算
6. **算子值时间传递**: 每步结束后更新 prev ← star 供下一步使用
7. **维度安全**: 添加数组边界检查和维度验证
8. **内存管理**: 实例变量存储算子值实现时间步间复用
9. **错误处理**: 仅预测步需要雅可比矩阵构建，校正步纯显式计算

## 关键数据维度

| 变量 | 维度 | 说明 |
|------|------|------|
| `U_seg` | `List[np.ndarray(n_points, n_eqs)]` | 各segment的解值 |
| `coeffs` | `np.ndarray(ns, n_eqs, dgN)` | 系数矩阵 |
| `L1_ops[eq_idx]` | `np.ndarray(n_points, dgN)` | L1算子矩阵 |
| `L1_beta[seg_idx][eq_idx]` | `np.ndarray(n_points,)` | L1算子值 |
| `F_mid` | `np.ndarray(n_points, n_eqs)` | 中点处F函数值 |
| `N_mid` | `np.ndarray(n_points, n_eqs)` | 中点处N函数值 |

## 数据流程图

```
时间步 n-1:  预测步 → L1*β^{n-1}, L2*β^{n-1} (存储)
                ↓
时间步 n:    预测步 → L1*β^*, L2*β^* (当前计算)
                ↓
            校正步 → 使用 L1*β^*(当前) + L1*β^{n-1}(预存)
                   → 使用 L2*β^*(当前) + L2*β^{n-1}(预存)
                ↓
            更新 → L1*β^* → L1_beta_prev (供下一步使用)
                 → L2*β^* → L2_beta_prev (供下一步使用)
```

## 性能优势

相比传统方法，该实现具有以下优势：

1. **减少矩阵运算**: 校正步完全避免矩阵求解和矩阵-向量乘法
2. **算子值时间复用**: 每个算子值计算一次，在两个时间步中使用
3. **内存局部性**: 算子值连续存储，提高缓存效率
4. **条件执行**: 第一步直接使用预测值，避免不必要的校正计算
5. **计算复杂度**: 从 O(2*矩阵运算) 降低到 O(1*矩阵运算 + 矢量运算)
