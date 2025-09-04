# IMEX-RK(2,2,2) 时间积分格式

## 一般形式（A、B算子表示）

**阶段1：**
$$\mathbf{U}^{(1)} = \mathbf{U}^n + \gamma \Delta t \cdot \mathbf{B}(\mathbf{U}^{(1)}) \cdot \mathbf{A}(\mathbf{U}^n)$$

**阶段2：**
$$\mathbf{U}^{(2)} = \mathbf{U}^n + (1-2\gamma) \Delta t \cdot \mathbf{B}(\mathbf{U}^{(1)}) \cdot \mathbf{A}(\mathbf{U}^n) + \gamma \Delta t \cdot \mathbf{B}(\mathbf{U}^{(2)}) \cdot \mathbf{A}(\mathbf{U}^{(1)})$$

**最终更新：**
$$\mathbf{U}^{n+1} = \mathbf{U}^n + \frac{\Delta t}{2} \left[\mathbf{B}(\mathbf{U}^{(1)}) \cdot \mathbf{A}(\mathbf{U}^n) + \mathbf{B}(\mathbf{U}^{(2)}) \cdot \mathbf{A}(\mathbf{U}^{(2)})\right]$$

## 具体问题设定

考虑时间相关偏微分方程：
$$\frac{\partial U}{\partial t} = L_1(U) + L_2(U) \odot F(U) + N(U)$$

其中：
- $L_1(U)$: 隐式线性算子（如扩散项）
- $L_2(U) \odot F(U)$: 隐式线性算子与非线性系数的逐点乘积
- $N(U)$: 显式非线性算子（如反应项）
- $\gamma = \frac{2-\sqrt{2}}{2} \approx 0.2929$

### 阶段1
$$U^{(1)} = U^n + \gamma \Delta t (L_1(U^{(1)}) + L_2(U^{(1)}) \odot F(U^n) + N(U^n))$$

### 阶段2  
$$U^{(2)} = U^n + (1-2\gamma) \Delta t (L_1(U^{(1)}) + L_2(U^{(1)}) \odot F(U^n) + N(U^n)) + \gamma \Delta t (L_1(U^{(2)}) + L_2(U^{(2)}) \odot F(U^{(1)}) + N(U^{(1)}))$$

### 最终更新（复用中间结果）

定义阶段算子值：
- $K^{(1)} = L_1(U^{(1)}) + L_2(U^{(1)}) \odot F(U^n) + N(U^n)$
- $K^{(2)} = L_1(U^{(2)}) + L_2(U^{(2)}) \odot F(U^{(2)}) + N(U^{(2)})$

则最终更新为：
$$U^{n+1} = U^n + \frac{\Delta t}{2} \left[ K^{(1)} + K^{(2)} \right]$$

**注意**：$K^{(1)}$ 已在阶段1求解过程中计算，$K^{(2)}$ 在阶段2求解过程中计算，可直接复用。

## 线性化求解形式

### 基础离散化
设 $U = V \boldsymbol{\beta}$，其中：
- $V$：特征矩阵（多项式基函数）
- $\boldsymbol{\beta}$：待求系数向量
- $L_1(U) = L_1 \boldsymbol{\beta}$，$L_1$：预编译线性算子矩阵
- $L_2(U) = L_2 \boldsymbol{\beta}$，$L_2$：预编译线性算子矩阵

### 阶段1的线性系统
将阶段1重新整理为隐式形式：
$$U^{(1)} - \gamma \Delta t L_1(U^{(1)}) - \gamma \Delta t L_2(U^{(1)}) \odot F(U^n) = U^n + \gamma \Delta t N(U^n)$$

代入离散化形式：
$$[V - \gamma \Delta t L_1 - \gamma \Delta t \text{diag}(F(U^n)) L_2] \boldsymbol{\beta}^{(1)} = U^n + \gamma \Delta t N(U^n)$$

### 阶段2的线性系统
将阶段2重新整理为隐式形式：
$$U^{(2)} - \gamma \Delta t L_1(U^{(2)}) - \gamma \Delta t L_2(U^{(2)}) \odot F(U^{(1)}) = \text{RHS}_2$$

其中右端项：
$$\text{RHS}_2 = U^n + (1-2\gamma) \Delta t [L_1(U^{(1)}) + L_2(U^{(1)}) \odot F(U^n)] + (1-\gamma) \Delta t N(U^{(1)})$$

代入离散化形式：
$$[V - \gamma \Delta t L_1 - \gamma \Delta t \text{diag}(F(U^{(1)})) L_2] \boldsymbol{\beta}^{(2)} = \text{RHS}_2$$

## 矩阵系统构造

### 系统矩阵形式
每个阶段需要求解线性系统：$A \boldsymbol{\beta} = \boldsymbol{b}$

**阶段1系统矩阵：**
$$A_1 = V - \gamma \Delta t L_1 - \gamma \Delta t \text{diag}(F(U^n)) L_2$$

**阶段2系统矩阵：**
$$A_2 = V - \gamma \Delta t L_1 - \gamma \Delta t \text{diag}(F(U^{(1)})) L_2$$

### 多方程系统
对于 $n_e$ 个方程的系统，矩阵结构为：
$$A_{final} = \begin{pmatrix}
A_{11} & 0 & \cdots & 0 \\
0 & A_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & A_{n_e n_e}
\end{pmatrix}$$

其中每个子块 $A_{ii}$：
$$A_{ii} = V - \gamma \Delta t L_{1,i} - \gamma \Delta t \text{diag}(F_i) L_{2,i}$$

### 右端向量
**阶段1右端向量：**
$$\boldsymbol{b}_1 = U^n + \gamma \Delta t N(U^n)$$

**阶段2右端向量：**
$$\boldsymbol{b}_2 = U^n + (1-2\gamma) \Delta t [L_1 \boldsymbol{\beta}^{(1)} + \text{diag}(F(U^n)) L_2 \boldsymbol{\beta}^{(1)}] + (1-\gamma) \Delta t N(U^{(1)})$$

## 实现细节

### 段级操作
在实际实现中，对每个段（segment）$s = 1, \ldots, n_s$：

**维度信息：**
- $U_s$: $(n_{points}, n_e)$ - 段级解值
- $\boldsymbol{\beta}_s$: $(n_e, d_{gN})$ - 段级系数矩阵
- $V_s$: $(n_{points}, d_{gN})$ - 段级特征矩阵
- $L_{1,s}$, $L_{2,s}$: $(n_{points}, d_{gN})$ - 段级算子矩阵

**段级线性系统：**
$$[V_s - \gamma \Delta t L_{1,s} - \gamma \Delta t \text{diag}(F_s) L_{2,s}] \boldsymbol{\beta}_s = \boldsymbol{b}_s$$

### 组装求解流程

#### 数据结构
- **全局解**：$U^n \in \mathbb{R}^{N_{total} \times n_e}$
- **分段解**：$\{U_s^n\}_{s=1}^{n_s}$，其中 $U_s^n \in \mathbb{R}^{n_{points,s} \times n_e}$
- **全局系数**：$\boldsymbol{\beta} \in \mathbb{R}^{n_s \times n_e \times d_{gN}}$ （作为整体求解）

#### 阶段1：组装系统求解
```
1. 分段准备：
   对于每个段 s = 1, ..., n_s：
       - 提取段解：U_s^n = extract_segment(U^n, s)
       - 构造段级雅可比矩阵：A_{1,s} = V_s - γΔt L_{1,s} - γΔt diag(F_s^n) L_{2,s}
       - 构造段级右端向量：b_{1,s} = U_s^n + γΔt N_s^n

2. 系统组装：
   - 组装全局雅可比矩阵：A_1 = block_diag(A_{1,1}, A_{1,2}, ..., A_{1,n_s})
   - 组装全局右端向量：b_1 = [b_{1,1}; b_{1,2}; ...; b_{1,n_s}]

3. 统一求解：
   - 求解全局系统：β^{(1)} = solve_linear_system(A_1, b_1)
   - 从全局系数提取分段系数：β_s^{(1)} = β^{(1)}[s, :, :]

4. 构造分段解：
   对于每个段 s = 1, ..., n_s：
       - U_s^{(1)} = V_s β_s^{(1)}
       - K_s^{(1)} = L_{1,s}β_s^{(1)} + diag(F_s^n)L_{2,s}β_s^{(1)} + N_s^n
```

#### 阶段2：使用阶段1结果组装求解
```
1. 分段准备（使用阶段1结果）：
   对于每个段 s = 1, ..., n_s：
       - 计算新F值：F_s^{(1)} = F(U_s^{(1)})
       - 构造段级雅可比矩阵：A_{2,s} = V_s - γΔt L_{1,s} - γΔt diag(F_s^{(1)}) L_{2,s}
       - 构造段级右端向量：b_{2,s} = RHS_{2,s}（包含阶段1的显式贡献）

2. 系统组装和求解：
   - 组装全局系统：A_2, b_2
   - 求解：β^{(2)} = solve_linear_system(A_2, b_2)
   - 提取分段系数：β_s^{(2)} = β^{(2)}[s, :, :]

3. 构造解和算子值：
   对于每个段 s = 1, ..., n_s：
       - U_s^{(2)} = V_s β_s^{(2)}
       - K_s^{(2)} = L_{1,s}β_s^{(2)} + diag(F_s^{(2)})L_{2,s}β_s^{(2)} + N_s^{(2)}
```

#### 最终更新：直接使用算子值
```
对于每个段 s = 1, ..., n_s：
    U_s^{n+1} = U_s^n + Δt/2 [K_s^{(1)} + K_s^{(2)}]

全局组装：
    U^{n+1} = assemble_global({U_s^{n+1}}_{s=1}^{n_s})
```

## 计算优化

### 中间结果复用
在阶段求解过程中，系统矩阵的构造需要计算算子贡献。这些贡献值可以在最终更新中直接复用：

**阶段1求解时计算：**
- $K^{(1)} = L_1 \boldsymbol{\beta}^{(1)} + \text{diag}(F(U^n)) L_2 \boldsymbol{\beta}^{(1)} + N(U^n)$

**阶段2求解时计算：**  
- $K^{(2)} = L_1 \boldsymbol{\beta}^{(2)} + \text{diag}(F(U^{(2)})) L_2 \boldsymbol{\beta}^{(2)} + N(U^{(2)})$

### 避免的重复计算
- **算子应用**：$L_1, L_2$ 算子不需要重新计算
- **非线性函数评估**：$N, F$ 函数值可以复用  
- **矩阵乘法**：算子与系数的乘积可以复用

### 组装求解的数据流

#### 关键计算步骤
1. **阶段1系统组装**：
   - 分段构造雅可比矩阵和右端向量
   - 组装成全局块对角系统
   - 统一求解全局系数 $\boldsymbol{\beta}^{(1)}$

2. **阶段1结果处理**：
   - 从全局系数提取分段系数
   - 构造分段解 $U_s^{(1)}$ 和算子值 $K_s^{(1)}$

3. **阶段2系统组装**：
   - 使用 $U_s^{(1)}$ 更新非线性项 $F_s^{(1)}$
   - 重新构造雅可比矩阵（非线性系数已更新）
   - 组装并求解第二个全局系统

4. **最终更新**：
   - 直接使用预计算的算子值进行加权平均
   - 分段更新后全局组装

#### 数据传递模式
```
时间步开始：
    全局解 U^n → {段解 U_s^n}_{s=1}^{n_s}  [拆分]

阶段1：
    {U_s^n}_{s=1}^{n_s} → 构造{A_{1,s}, b_{1,s}}_{s=1}^{n_s} → 组装A_1, b_1 → 求解β^{(1)} → 提取{β_s^{(1)}}_{s=1}^{n_s}

阶段1后处理：
    {β_s^{(1)}}_{s=1}^{n_s} → {U_s^{(1)}, K_s^{(1)}}_{s=1}^{n_s}

阶段2：
    {U_s^{(1)}}_{s=1}^{n_s} → 构造{A_{2,s}, b_{2,s}}_{s=1}^{n_s} → 组装A_2, b_2 → 求解β^{(2)} → 提取{β_s^{(2)}}_{s=1}^{n_s}

阶段2后处理：
    {β_s^{(2)}}_{s=1}^{n_s} → {U_s^{(2)}, K_s^{(2)}}_{s=1}^{n_s}

最终更新：
    {K_s^{(1)}, K_s^{(2)}}_{s=1}^{n_s} → {U_s^{n+1}}_{s=1}^{n_s} → 全局解U^{n+1}
```

### 计算复杂度和存储
**线性求解器调用**：每个时间步需要2次大型线性系统求解
- 全局系统规模：$(n_s \times n_e \times n_{points}) \times (n_s \times n_e \times d_{gN})$
- 块对角结构可以利用进行优化

**存储需求**：
- 全局系数：$\boldsymbol{\beta}^{(1)}, \boldsymbol{\beta}^{(2)} \in \mathbb{R}^{n_s \times n_e \times d_{gN}}$
- 分段解值：$\{U_s^n, U_s^{(1)}, U_s^{(2)}\}$ 和算子值 $\{K_s^{(1)}, K_s^{(2)}\}$
- 临时组装矩阵：全局雅可比矩阵和右端向量

## Butcher表

### 隐式部分 ($A_{imp}$)
$$A_{imp} = \begin{pmatrix} \gamma & 0 \\ 1-2\gamma & \gamma \end{pmatrix}$$

### 显式部分 ($A_{exp}$)
$$A_{exp} = \begin{pmatrix} 0 & 0 \\ 1-\gamma & 0 \end{pmatrix}$$

### 权重向量
$$b = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}$$

## 稳定性特性
- 隐式项（$L_1, L_2 \odot F$）：A-稳定
- 显式项（$N$）：二阶精度显式处理
- 总体精度：二阶

## 代码优化建议

### 1. 算子值复用优化

#### 当前问题
在 `_imex_final_update_U_seg()` 中重复计算算子值：
```python
# 当前实现每次都重新计算
for U_seg_stage, coeffs_stage, weight in stage_data:
    # 重复计算 L1, L2⊙F, N
    L1_contrib = L1_seg @ beta_seg_stage
    L2F_contrib = np.diag(F_eq) @ L2_seg @ beta_seg_stage
    N_vals = self.fitter.N_func(features, U_seg_stage)
```

#### 优化建议
**方案A：在阶段求解时存储算子值**
```python
# 修改 _solve_imex_stage_U_seg 返回算子值
def _solve_imex_stage_U_seg(self, **kwargs) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]:
    # ... 现有代码 ...
    
    # 计算并存储算子值
    K_stage_segments = []
    for segment_idx in range(self.fitter.ns):
        K_seg = self._compute_stage_operator_values(
            segment_idx, coeffs_stage, U_stage_segments[segment_idx], **kwargs
        )
        K_stage_segments.append(K_seg)
    
    return U_stage_segments, coeffs_stage, K_stage_segments
```

**方案B：缓存中间结果**
```python
# 在类中添加缓存属性
self._stage_operator_cache = {}

def _compute_stage_operator_values(self, segment_idx, coeffs, U_seg, stage):
    cache_key = f"stage_{stage}_seg_{segment_idx}"
    if cache_key not in self._stage_operator_cache:
        # 计算算子值
        K_seg = self._calculate_operators(segment_idx, coeffs, U_seg)
        self._stage_operator_cache[cache_key] = K_seg
    return self._stage_operator_cache[cache_key]
```

### 2. 内存管理优化

#### 当前问题
- 临时数组频繁创建：`stage_contribution = np.zeros_like(U_n_seg_current)`
- 维度转换开销：`L1_contrib.reshape(-1, 1)`
- F函数重复调用：每个方程都调用 `self.fitter.F_func()`

#### 优化建议
```python
def _imex_final_update_U_seg_optimized(self, U_n_seg, U_1_seg, U_2_seg, dt, coeffs_1, coeffs_2):
    ne = self.config.n_eqs
    U_seg_new = []
    
    # 预分配缓冲区
    temp_buffer = None
    
    for segment_idx in range(self.fitter.ns):
        n_points = len(self.fitter.data["x_segments_norm"][segment_idx])
        U_n_seg_current = U_n_seg[segment_idx]
        
        # 复用缓冲区
        if temp_buffer is None or temp_buffer.shape != U_n_seg_current.shape:
            temp_buffer = np.zeros_like(U_n_seg_current)
        else:
            temp_buffer.fill(0.0)
        
        # 直接使用预计算的算子值
        if hasattr(self, '_K_stage1') and hasattr(self, '_K_stage2'):
            K1_seg = self._K_stage1[segment_idx]
            K2_seg = self._K_stage2[segment_idx]
            
            # 直接加权求和
            temp_buffer[:] = U_n_seg_current + dt * 0.5 * (K1_seg + K2_seg)
        else:
            # 回退到当前实现
            temp_buffer = self._compute_final_update_fallback(...)
        
        U_seg_new.append(temp_buffer.copy())
    
    return U_seg_new
```

### 3. 计算效率改进

#### 矩阵运算优化
```python
# 当前：多次矩阵乘法
L1_contrib = L1_seg @ beta_seg_stage
L2F_contrib = np.diag(F_eq) @ L2_seg @ beta_seg_stage

# 优化：减少临时矩阵创建
def _optimized_operator_application(self, L1_seg, L2_seg, F_eq, beta_seg):
    # 直接计算，避免创建对角矩阵
    L2_beta = L2_seg @ beta_seg
    L2F_contrib = L2_beta * F_eq  # 逐元素乘法更高效
    
    L1_contrib = L1_seg @ beta_seg
    
    return L1_contrib, L2F_contrib
```

#### F函数调用优化
```python
# 当前：每个方程单独调用F函数
for eq_idx in range(ne):
    F_vals = self.fitter.F_func(features, U_seg_stage)
    F_eq = F_vals[:, eq_idx]

# 优化：批量计算F值
F_vals_all = self.fitter.F_func(features, U_seg_stage)  # 只调用一次
for eq_idx in range(ne):
    F_eq = F_vals_all[:, eq_idx]
```

### 4. 错误处理和调试优化

#### 当前问题
- 硬编码的调试打印：`print(f"Debug stage_contribution.shape={stage_contribution.shape}")`
- 错误处理不完整
- 代码中有未使用的变量

#### 优化建议
```python
import logging

class ImexRK222(BaseTimeScheme):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
    def _imex_final_update_U_seg(self, **kwargs):
        try:
            # 移除调试打印，使用日志
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Starting final update for {len(U_n_seg)} segments")
            
            # ... 优化后的实现 ...
            
        except Exception as e:
            self.logger.error(f"Error in final update: {e}")
            raise
        finally:
            # 清理缓存（如果需要）
            if hasattr(self, '_stage_operator_cache'):
                self._stage_operator_cache.clear()
```

### 5. 并行化建议

#### 段级并行
```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def _parallel_segment_processing(self, segment_func, *args, **kwargs):
    """并行处理所有段"""
    if self.fitter.ns > 4:  # 只在段数较多时使用并行
        with ThreadPoolExecutor(max_workers=min(4, self.fitter.ns)) as executor:
            futures = []
            for segment_idx in range(self.fitter.ns):
                future = executor.submit(segment_func, segment_idx, *args, **kwargs)
                futures.append(future)
            
            results = [future.result() for future in futures]
            return results
    else:
        # 串行处理小规模问题
        return [segment_func(i, *args, **kwargs) for i in range(self.fitter.ns)]
```

### 6. 数值稳定性改进

#### gamma参数处理
```python
# 当前：硬编码或注释掉的gamma值
self.gamma = (2 - np.sqrt(2)) / 2  # ≈ 0.2928932
#self.gamma = 1

# 优化：参数验证和自适应
def __init__(self, config):
    super().__init__(config)
    
    # 验证和设置gamma
    gamma_from_config = getattr(config, 'imex_gamma', None)
    if gamma_from_config is not None:
        self.gamma = gamma_from_config
    else:
        self.gamma = (2 - np.sqrt(2)) / 2
    
    # 验证gamma在有效范围内
    if not (0 < self.gamma < 1):
        raise ValueError(f"IMEX gamma parameter must be in (0,1), got {self.gamma}")
```

### 7. 接口优化

#### 返回值一致性
```python
# 当前：time_step返回不一致的结果
return (U_new, U_seg_stage1, coeffs_stage1)  # 应该返回最终结果

# 优化：返回完整和一致的结果
def time_step(self, U_n, U_seg, dt, coeffs_n=None, current_time=0.0, step=0):
    # ... 计算过程 ...
    
    # 转换回全局数组使用最终结果而不是stage1
    U_new = self.fitter.segments_to_global(U_seg_new)
    
    return (
        U_new,          # 全局最终解
        U_seg_new,      # 段级最终解
        coeffs_stage2,  # 最终系数
    )
```

### 8. 关键性能问题修复

#### 问题1：最终更新中的重复计算
```python
# 当前代码在 _imex_final_update_U_seg 中重新计算所有算子值
# 优化：直接在阶段求解时存储K值

def _solve_imex_stage_U_seg(self, **kwargs):
    # ... 现有求解代码 ...
    
    # 在求解后立即计算并存储K值
    K_stage_segments = self._compute_and_store_K_values(
        U_stage_segments, coeffs_stage, kwargs.get("stage")
    )
    
    # 存储到实例变量
    setattr(self, f'_K_stage_{kwargs.get("stage")}', K_stage_segments)
    
    return U_stage_segments, coeffs_stage

def _imex_final_update_U_seg_fast(self, U_n_seg, dt, **kwargs):
    """优化后的最终更新 - 直接使用预计算的K值"""
    U_seg_new = []
    
    for segment_idx in range(self.fitter.ns):
        U_n_current = U_n_seg[segment_idx]
        K1 = self._K_stage_1[segment_idx]
        K2 = self._K_stage_2[segment_idx]
        
        # 直接计算最终更新
        U_new_seg = U_n_current + dt * 0.5 * (K1 + K2)
        U_seg_new.append(U_new_seg)
    
    return U_seg_new
```

#### 问题2：gamma系数处理不一致
```python
# 在build_stage_jacobian中出现的问题
if stage == 2 and step > 0:
    J_eq -= 0.5* dt * L1  # 应该是 gamma * dt
else:
    J_eq -= dt * L1       # 应该是 gamma * dt

# 修复：统一使用gamma参数
J_eq -= self.gamma * dt * L1
```

#### 问题3：内存泄漏预防
```python
def __del__(self):
    """清理缓存防止内存泄漏"""
    if hasattr(self, '_stage_operator_cache'):
        self._stage_operator_cache.clear()
    if hasattr(self, '_K_stage_1'):
        delattr(self, '_K_stage_1')
    if hasattr(self, '_K_stage_2'):
        delattr(self, '_K_stage_2')
```

