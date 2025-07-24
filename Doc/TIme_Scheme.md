# IMEX Runge-Kutta 时间积分方案

## 1. 问题形式

### 1.1 单方程形式
$$\frac{\partial u}{\partial t} = L_1(u) + N(u) + L_2(u) F(u)$$

### 1.2 方程组形式
对于包含 $m$ 个变量 $\mathbf{u} = [u_1, u_2, \ldots, u_m]^T$ 的方程组系统：

$$\frac{\partial \mathbf{u}}{\partial t} = \mathbf{L}_1(\mathbf{u}) + \mathbf{N}(\mathbf{u}) + \mathbf{L}_2(\mathbf{u}) \mathbf{F}(\mathbf{u})$$

### 1.3 算子处理策略
- **$\mathbf{L}_1, \mathbf{L}_2$**：**隐式处理**（线性算子）
  - 通常包含扩散项、高阶导数项
  - 需要隐式处理以保证数值稳定性
- **$\mathbf{N}, \mathbf{F}$**：**显式处理**（非线性项）
  - 完全非线性项
  - 显式处理以提高计算效率

## 1.5 一般Runge-Kutta方法回顾

### 1.5.1 标准Runge-Kutta方法
对于常微分方程 $\frac{dy}{dt} = f(t,y)$，$s$阶段的Runge-Kutta方法的一般形式为：

**阶段计算：**
$$k_i = f\left(t_n + c_i \Delta t, y_n + \Delta t \sum_{j=1}^s a_{ij} k_j\right), \quad i=1,\ldots,s$$

**时间步更新：**
$$y_{n+1} = y_n + \Delta t \sum_{i=1}^s b_i k_i$$

其中Butcher表为：
$$
\begin{array}{c|cccc}
c_1 & a_{11} & a_{12} & \cdots & a_{1s} \\
c_2 & a_{21} & a_{22} & \cdots & a_{2s} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
c_s & a_{s1} & a_{s2} & \cdots & a_{ss} \\
\hline
 & b_1 & b_2 & \cdots & b_s
\end{array}
$$

### 1.5.2 显式vs隐式RK方法
- **显式RK(ERK)**：当 $j \geq i$ 时 $a_{ij} = 0$，计算简单但稳定性受限
- **隐式RK(IRK)**：允许 $a_{ij} \neq 0$，需要求解方程组但稳定性好
- **对角隐式RK(DIRK)**：当 $j > i$ 时 $a_{ij} = 0$，平衡了计算效率和稳定性

### 1.5.3 分裂方法基础
对于右端项分裂为 $f(t,y) = f_E(t,y) + f_I(t,y)$ 的系统：
- **显式处理**：$f_E$ 用显式方法处理
- **隐式处理**：$f_I$ 用隐式方法处理

## 2. 通用 IMEX-RK 框架

### 2.1 Butcher 表结构

对于 $s$-阶段IMEX-RK方法，定义两套Butcher表：

**隐式表 (Implicit Tableau):**
$$\tilde{A} = (\tilde{a}_{ij})_{s \times s}, \quad \tilde{b} = (\tilde{b}_i)_{s \times 1}, \quad \tilde{c} = (\tilde{c}_i)_{s \times 1}$$

**显式表 (Explicit Tableau):**
$$A = (a_{ij})_{s \times s}, \quad b = (b_i)_{s \times 1}, \quad c = (c_i)_{s \times 1}$$

通常满足：$c = \tilde{c}$ 和 $b = \tilde{b}$

### 2.2 阶段计算

对于第 $i$ 阶段 ($i = 1, 2, \ldots, s$)：

**步骤1：构造右端项**
$$\begin{aligned}
\mathbf{RHS}^{(i)} = \boldsymbol{u}^n &+ \Delta t \sum_{j=1}^{i-1} a_{ij} \mathbf{N}(\mathbf{u}^{(j)}) \\
&+ \Delta t \sum_{j=1}^{i-1} \tilde{a}_{ij} \left[\mathbf{L}_1 \boldsymbol{\beta}^{(j)} + \mathbf{L}_2 \odot \mathbf{F}(\mathbf{u}^{(j)}) \boldsymbol{\beta}^{(j)}\right]
\end{aligned}$$

$$\mathbf{L} \odot \mathbf{F} = \mathbf{L}_{ijk}\cdot \mathbf{F}_{ij}$$
**步骤2：求解线性系统**
$$\left[\mathbf{V} - \Delta t \tilde{a}_{ii} \mathbf{L}_1 - \Delta t \tilde{a}_{ii} \mathbf{L}_2 \odot \mathbf{F}(\mathbf{u}^{prev})\right] \boldsymbol{\beta}^{(i)} = \mathbf{RHS}^{(i)}$$

**步骤3：计算阶段解**
$$\mathbf{u}^{(i)} = \mathbf{V} \boldsymbol{\beta}^{(i)}$$

其中 $\mathbf{u}^{prev}$ 是前一阶段值（或初始解 $\mathbf{u}^n = \mathbf{V} \boldsymbol{\beta}^n$）

### 2.3 时间步更新
$$\boldsymbol{\beta}^{n+1} = \boldsymbol{\beta}^n + \Delta t \sum_{i=1}^{s} b_i \left[\mathbf{L}_1 \boldsymbol{\beta}^{(i)} + \mathbf{N}(\mathbf{u}^{(i)}) + \mathbf{L}_2 \odot \mathbf{F}(\mathbf{u}^{(i)}) \boldsymbol{\beta}^{(i)}\right]$$

**最终解：**
$$\mathbf{u}^{n+1} = \mathbf{V} \boldsymbol{\beta}^{n+1}$$

## 3. IMEX-RK(2,2,2) 方法

### 3.1 参数
$$\gamma = \frac{2-\sqrt{2}}{2} \approx 0.2928932$$

### 3.2 Butcher 表

**隐式表：**
$$
\begin{array}{c|cc}
\gamma & \gamma & 0 \\
1 & 1-2\gamma & \gamma \\
\hline
 & \frac{1}{2} & \frac{1}{2}
\end{array}
$$

**显式表：**
$$
\begin{array}{c|cc}
0 & 0 & 0 \\
1-\gamma & 1-\gamma & 0 \\
\hline
 & \frac{1}{2} & \frac{1}{2}
\end{array}
$$

### 3.3 阶段计算

**阶段 1：**
$$\left[\mathbf{V} - \gamma \Delta t \mathbf{L}_1 - \gamma \Delta t \mathbf{L}_2 \odot \mathbf{F}(\mathbf{u}^n)\right] \boldsymbol{\beta}^{(1)} = \mathbf{u}^n + \gamma\Delta t \mathbf{N}(\mathbf{u}^n)$$
$$\mathbf{u}^{(1)} = \mathbf{V} \boldsymbol{\beta}^{(1)}$$


**阶段 2：**
$$\begin{aligned}
\left[\mathbf{V} - \gamma \Delta t \mathbf{L}_1 - \gamma \Delta t \mathbf{L}_2 \odot \mathbf{F}(\mathbf{u}^{(1)})\right] \boldsymbol{\beta}^{(2)} = \boldsymbol{u}^n &+ \Delta t(1-2\gamma) \left[\mathbf{L}_1 + \mathbf{L}_2 \odot \mathbf{F}(\mathbf{u}^{(1)}) \right] \boldsymbol{\beta}^{(1)}\\
&+ \Delta t(1-\gamma) \mathbf{N}(\mathbf{u}^{(1)})
\end{aligned}$$
$$\mathbf{u}^{(2)} = \mathbf{V} \boldsymbol{\beta}^{(2)}$$

**说明：**
- $\mathbf{L}_1, \mathbf{L}_2$：预计算的线性算子矩阵，作用于基函数$\mathbf{V}$
- $\boldsymbol{\beta}^n$：第n时间步的基函数系数向量
- $\mathbf{N}(\mathbf{u})$：非线性项$\mathbf{N}(\mathbf{u})$在基函数空间的投影
- $\mathbf{F}(\mathbf{u})$：$\mathbf{F}(\mathbf{u})$构成的对角矩阵
### 3.4 时间步更新
$$
\mathbf{u}^{n+1} = \mathbf{u}^n + \frac{\Delta t}{2} \left[
\mathbf{L}_1 \boldsymbol{\beta}^{(1)} + \mathbf{L}_1 \boldsymbol{\beta}^{(2)} + \mathbf{N}(\mathbf{u}^{(1)}) + \mathbf{N}(\mathbf{u}^{(2)}) + \mathbf{L}_2 \odot \mathbf{F}(\mathbf{u}^{(1)}) \boldsymbol{\beta}^{(1)} + \mathbf{L}_2 \odot \mathbf{F}(\mathbf{u}^{(2)}) \boldsymbol{\beta}^{(2)}
\right]
$$

**最终解：**
$$\mathbf{u}^{n+1} = \mathbf{V} \boldsymbol{\beta}^{n+1}$$

## 4. 应用示例

### 4.1 Allen-Cahn 方程
$$\frac{\partial u}{\partial t} = \epsilon^2 \nabla^2 u - u(u^2 - 1)$$

**算子分离:**
- $L_1(u) = \epsilon^2 \nabla^2 u$
- $N(u) = -u(u^2 - 1)$
- $L_2(u) = 0$, $F(u) = 0$

**阶段方程:**

*阶段1:*
$$\left(I - \gamma \Delta t \epsilon^2 \nabla^2\right) U_1 = u^n$$

*阶段2:*
$$\left(I - \gamma \Delta t \epsilon^2 \nabla^2\right) U_2 = u^n + \Delta t(1-2\gamma) \epsilon^2 \nabla^2 U_1 + \Delta t(1-\gamma) [-U_1((U_1)^2 - 1)]$$

### 4.2 Cahn-Hilliard 方程
$$\frac{\partial u}{\partial t} = -\nabla^2(\epsilon^2 \nabla^2 u - u^3 + u)$$

**重写为:** $u_t = -\epsilon^2 \nabla^4 u - \nabla^2 u + \nabla^2 u \cdot u^2$

**算子分离:**
- $L_1(u) = -\epsilon^2 \nabla^4 u - \nabla^2 u$
- $N(u) = 0$
- $L_2(u) = \nabla^2 u$, $F(u) = u^2$

### 4.3 对流-扩散-反应方程
$$\frac{\partial u}{\partial t} = \epsilon \nabla^2 u + \nabla \cdot (v u) - \alpha u^3$$

**算子分离:**
- $L_1(u) = \epsilon \nabla^2 u$
- $N(u) = -\alpha u^3$
- $L_2(u) = \nabla \cdot (v u)$, $F(u) = 1$

### 4.4 反应-扩散系统
$$\begin{align}
\frac{\partial u}{\partial t} &= D_u \nabla^2 u + f(u,v) \\
\frac{\partial v}{\partial t} &= D_v \nabla^2 v + g(u,v)
\end{align}$$

**算子分离:**
- $\mathbf{L_1}(\mathbf{u}) = \begin{pmatrix} D_u \nabla^2 u \\ D_v \nabla^2 v \end{pmatrix}$
- $\mathbf{N}(\mathbf{u}) = \begin{pmatrix} f(u,v) \\ g(u,v) \end{pmatrix}$
- $\mathbf{L_2} = \mathbf{0}$, $\mathbf{F} = \mathbf{0}$

## 5. 实现考虑

### 5.1 矩阵组装

对于每个阶段，需要构建并求解线性系统：
$$\mathbf{A}^{(i)} \mathbf{U}^{(i)} = \mathbf{b}^{(i)}$$

其中系统矩阵为：
$$\mathbf{A}^{(i)} = \mathbf{I} - \Delta t \tilde{a}_{ii} \mathbf{L}_1 - \Delta t \tilde{a}_{ii} \mathbf{L}_2 \text{diag}(\mathbf{F}(\mathbf{U}^{prev}))$$

### 5.2 计算效率

**DIRK特性:** 对于IMEX-RK(2,2,2)，两个阶段的系统矩阵具有相同的结构（对角元素都是$\gamma$），可以重复使用矩阵分解。

**稀疏性:** 利用偏微分方程离散化矩阵的稀疏特性，使用稀疏矩阵求解器。

### 5.3 稳定性分析

**稳定性约束:**
$$\Delta t \leq \frac{C_{stability}}{\max(\rho(\mathbf{L_1}), \rho(\mathbf{L_2}) \max|\mathbf{F}'(\mathbf{u})|, \max|\mathbf{N}'(\mathbf{u})|)}$$

其中：
- $\rho(\cdot)$ 表示矩阵的谱半径
- $C_{stability} \approx 1.5$ 对于IMEX-RK(2,2,2)

## 6. 其他 IMEX-RK 格式

### 6.1 IMEX-RK(1,1,1) - Backward Euler/Forward Euler
$$A = (0), \quad \tilde{A} = (1), \quad b = \tilde{b} = (1)$$

### 6.2 IMEX-RK(2,3,2) - ARS(2,3,2)
$$\gamma = \frac{3 + \sqrt{3}}{6}$$

### 6.3 IMEX-RK(3,4,3) - ARS(3,4,3)
高阶格式，适用于需要高精度的问题。

## 7. 算法实现框架

```python
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

def imex_rk_step(u_n, dt, A_exp, A_imp, b, L1, L2, N, F):
    """
    IMEX-RK 时间步进函数
    
    参数:
    u_n: 当前时间步解向量
    dt: 时间步长
    A_exp: 显式Butcher矩阵(s×s)
    A_imp: 隐式Butcher矩阵(s×s)
    b: 权重向量(长度s)
    L1: L1线性算子(函数或矩阵)
    L2: L2线性算子(函数或矩阵)
    N: 非线性项函数
    F: 非线性函数项
    """
    s = len(b)  # 阶段数
    n = len(u_n)  # 自由度数量
    U = [None] * s  # 存储各阶段解
    
    # 确保L1, L2是矩阵形式
    L1_mat = L1(u_n) if callable(L1) else L1
    L2_mat = L2(u_n) if callable(L2) else L2
    
    for i in range(s):  # 遍历每个阶段
        # 计算右端项RHS
        rhs = u_n.copy()
        
        # 显式项贡献 (j < i)
        for j in range(i):
            # 显式非线性项
            if A_exp[i, j] != 0:
                rhs += dt * A_exp[i, j] * N(U[j])
            
            # 隐式算子的显式部分 (j < i)
            if A_imp[i, j] != 0:
                # L1项
                L1_val = L1_mat @ U[j] if hasattr(L1_mat, 'dot') else L1_mat * U[j]
                # L2*F项 (使用当前阶段j的F值)
                F_j = F(U[j])
                L2F_val = L2_mat @ (F_j * U[j]) if hasattr(L2_mat, 'dot') else L2_mat * (F_j * U[j])
                
                rhs += dt * A_imp[i, j] * (L1_val + L2F_val)
        
        # 隐式处理 (对角线项)
        if A_imp[i, i] != 0:
            # 确定F的先前值
            if i == 0:
                F_prev = F(u_n)  # 第一阶段使用u_n
            else:
                F_prev = F(U[i-1])  # 后续阶段使用前一阶段值
            
            # 构建系统矩阵
            I = eye(n)
            alpha = dt * A_imp[i, i]
            
            # L1部分
            L1_term = alpha * L1_mat
            
            # L2*diag(F_prev)部分
            if isinstance(F_prev, np.ndarray):
                diag_F = diags(F_prev)
                L2_term = alpha * diag_F @ L2_mat
            else:  # 标量情况
                L2_term = alpha * F_prev * L2_mat
                
            A_matrix = I - L1_term - L2_term
            
            # 求解线性系统
            U[i] = spsolve(A_matrix, rhs)
        else:
            # 显式阶段
            U[i] = rhs
    
    # 最终更新
    u_new = u_n.copy()
    for i in range(s):
        # 计算当前阶段各项
        L1_val = L1_mat @ U[i] if hasattr(L1_mat, 'dot') else L1_mat * U[i]
        N_val = N(U[i])
        
        F_i = F(U[i])
        L2F_val = L2_mat @ (F_i * U[i]) if hasattr(L2_mat, 'dot') else L2_mat * (F_i * U[i])
        
        # 累加贡献
        u_new += dt * b[i] * (L1_val + N_val + L2F_val)
    
    return u_new
```

### 7.1 IMEX-RK(2,2,2) 特定实现

```python
def imex_rk22_step(u_n, dt, L1, L2, N, F):
    """IMEX-RK(2,2,2) 特定时间步进函数"""
    γ = (2 - np.sqrt(2)) / 2  # ≈0.2928932
    
    # 构建 Butcher 表
    A_imp = np.array([
        [γ, 0],
        [1 - 2*γ, γ]
    ])
    
    A_exp = np.array([
        [0, 0],
        [1 - γ, 0]
    ])
    
    b = np.array([0.5, 0.5])
    
    return imex_rk_step(u_n, dt, A_exp, A_imp, b, L1, L2, N, F)
```

### 7.2 应用示例：Allen-Cahn 方程

```python
# 参数设置
epsilon = 0.01
nx = 100  # 空间离散点数
dx = 1.0 / (nx - 1)

# 创建离散拉普拉斯算子 (1D)
def laplacian1d(n, dx):
    diag = -2 * np.ones(n) / dx**2
    off_diag = np.ones(n - 1) / dx**2
    return diags([off_diag, diag, off_diag], [-1, 0, 1])

L1_mat = epsilon**2 * laplacian1d(nx, dx)

# 定义非线性项
def N(u):
    return -u * (u**2 - 1)

# F项为零
def F(u):
    return 0

# 初始化
u0 = np.random.rand(nx)  # 初始条件
dt = 0.1 * dx**2  # 时间步长

# 时间步进
u_current = u0
for _ in range(100):
    u_next = imex_rk22_step(u_current, dt, L1_mat, 0, N, F)
    u_current = u_next
```

## 8. 关键特性

1. **稳定性**：隐式处理刚性项允许更大时间步长
2. **效率**：对角隐式RK(DIRK)特性使系统矩阵结构相同
3. **灵活性**：可处理多种算子组合：
   - 纯扩散 ($\mathbf{L}_1$)
   - 线性对流 ($\mathbf{L}_2$)
   - 非线性反应 ($\mathbf{N}$)
   - 非线性系数 ($\mathbf{F}$)

## 9. 程序实现方案

### 9.1 配置结构
```json
{
   "eq": "L_1(u) + N(u) + L_2(u)*F(u)",
   "eq_L1": "L_1(u)",
   "eq_N": "N(u)",
   "eq_L2": "L_2(u)",
   "eq_F": "F(u)",
   "var": {},
   "constants": {}
}
```

### 9.2 实现流程

1. **config.py** 中解析整体eq导数信息列表，根据导数列表生成L₁和L₂的导数列表和N和F的导数列表

2. **fitter.py** 中构造 L₁(U)和 L₂(U)算子矩阵，根据线性参数生成函数 L₁(U*β)和L₂(U*β)，以及函数 N(β)和 F(β)，基于分区组合。

3. **时间推进**：基于时间格式调用这些函数组成时间推进。

### 9.3 核心设计原则

1. 通过任意线性和非线性算子形成 $F(u)$ 函数
2. 通过任意线性算子形成 $L(U)$ 函数，其中 $u = \sum U \cdot \beta$，$U$ 是基函数。所以 $L(U)$ 是固定的，初始就可以生成。后续只需要更新 $\beta$
3. 根据config算子读入，建立算子函数

## 10. DeePoly框架下的IMEX-RK(2,2,2)实现

### 10.1 核心设计思想

DeePoly框架下的IMEX-RK(2,2,2)实现基于以下核心思想：

#### 10.1.1 算子预编译策略
- **线性算子预编译**：$\mathbf{L}_1, \mathbf{L}_2$ 算子矩阵在初始化阶段完全预编译和缓存
- **非线性算子函数化**：$\mathbf{N}, \mathbf{F}$ 算子编译为直接接受解值 $\mathbf{u}$ 的函数
- **分层预编译**：特征矩阵 → 线性算子 → 非线性函数 → 边界条件的分层预编译

#### 10.1.2 解值直接操作原则
- **时间循环解值操作**：时间推进过程中直接操作解值 $\mathbf{u}^n$，避免系数转换开销
- **算子直接调用**：$\mathbf{N}(\mathbf{u}), \mathbf{F}(\mathbf{u})$ 直接以解值为输入，符合物理直觉
- **最终系数转换**：仅在输出时进行解值到系数的转换，保持框架兼容性

#### 10.1.3 混合表示优势
- **计算效率**：减少频繁的系数-解值转换
- **数值稳定性**：直接解值操作避免累积转换误差
- **算子一致性**：所有算子统一使用解值接口

### 10.2 算法逻辑流程

#### 10.2.1 初始化阶段 (Precompilation Phase)
```
1. 特征矩阵生成和缓存
   for segment_idx in range(ns):
       features[segment_idx] = compute_features(segment_idx, model)
       
2. 线性算子预编译
   for segment_idx in range(ns):
       if has_L1: L1_operators[segment_idx] = L1_func(features[segment_idx])
       if has_L2: L2_operators[segment_idx] = L2_func(features[segment_idx])
       
3. 非线性函数预编译
   for segment_idx in range(ns):
       if has_N: N_functions[segment_idx] = compile_N_function(features[segment_idx])
       if has_F: F_functions[segment_idx] = compile_F_function(features[segment_idx])
       
4. 边界条件预编译
   A_constraints, b_constraints = compile_boundary_conditions(model)
```

#### 10.2.2 时间推进阶段 (Time Evolution Phase)
```
输入: u_current (当前时间步解值)
输出: u_new (新时间步解值)

1. 阶段1求解:
   U_stage1 = solve_imex_stage(u_current, None, dt, stage=1)
   
2. 阶段2求解:
   U_stage2 = solve_imex_stage(u_current, U_stage1, dt, stage=2)
   
3. 最终更新:
   u_new = imex_final_update(u_current, U_stage1, U_stage2, dt)
```

### 10.3 核心函数伪代码

#### 10.3.1 IMEX-RK时间步函数

```python
def imex_rk_time_step(u_n: ndarray, dt: float) -> ndarray:
    """
    执行IMEX-RK(2,2,2)时间步进 - 基于解值的操作
    
    输入: u_n - 当前时间步的解值 (全局拼接向量)
    输出: u_new - 新时间步的解值
    """
    γ = (2 - sqrt(2)) / 2  # IMEX-RK(2,2,2) 参数
    
    # 阶段1: 求解 U^(1) 
    U_stage1 = solve_imex_stage_u(u_n, None, dt, stage=1)
    
    # 阶段2: 求解 U^(2)
    U_stage2 = solve_imex_stage_u(u_n, U_stage1, dt, stage=2)
    
    # 最终更新: 计算 u^{n+1}
    u_new = imex_final_update_u(u_n, U_stage1, U_stage2, dt)
    
    return u_new
```

#### 10.3.2 阶段求解函数

```python
def solve_imex_stage_u(u_n: ndarray, u_prev: ndarray, dt: float, stage: int) -> ndarray:
    """
    求解IMEX-RK阶段 - 基于解值操作
    
    阶段1: [V - γΔt(L1 + L2⊙F(u^n))] β^(1) = u^n + γΔt N(u^n)
    阶段2: [V - γΔt(L1 + L2⊙F(u^(1)))] β^(2) = u^n + 显式项
    """
    
    # 准备阶段数据
    stage_data = {
        'u_n': u_n,
        'u_prev': u_prev, 
        'dt': dt,
        'stage': stage,
        'gamma': γ,
        'operation': 'imex_stage'
    }
    
    # 使用base_fitter的fit方法求解阶段系数
    coeffs_stage = fit(**stage_data)
    
    # 将系数转换为解值
    u_stage = construct_solution_from_coeffs(coeffs_stage)
    
    return u_stage
```

#### 10.3.3 最终更新函数

```python
def imex_final_update_u(u_n: ndarray, U_stage1: ndarray, U_stage2: ndarray, dt: float) -> ndarray:
    """
    IMEX-RK最终更新步骤 - 直接计算u^{n+1}
    
    根据公式3.4:
    u^{n+1} = u^n + Δt/2 * [L1(U^(1)) + L1(U^(2)) + N(U^(1)) + N(U^(2)) 
                           + L2⊙F(U^(1)) + L2⊙F(U^(2))]
    """
    
    total_contribution = zeros_like(u_n)
    b_weights = [0.5, 0.5]  # IMEX-RK(2,2,2) 权重
    
    for stage_idx, (U_stage, weight) in enumerate([(U_stage1, b_weights[0]), 
                                                   (U_stage2, b_weights[1])]):
        stage_contribution = zeros_like(u_n)
        
        # 按段计算贡献
        start_idx = 0
        for segment_idx in range(ns):
            n_points = len(x_segments[segment_idx])
            end_idx = start_idx + n_points
            
            u_seg = U_stage[start_idx:end_idx]
            
            # L1项贡献
            if has_L1:
                L1_seg = L1_operators[segment_idx]
                stage_contribution[start_idx:end_idx] += L1_seg @ u_seg
            
            # N项贡献 (非线性，直接以u值作为输入)
            if has_N:
                N_vals = N_functions[segment_idx](u_seg)
                stage_contribution[start_idx:end_idx] += N_vals
            
            # L2⊙F项贡献
            if has_L2 and has_F:
                L2_seg = L2_operators[segment_idx]
                F_vals = F_functions[segment_idx](u_seg)
                stage_contribution[start_idx:end_idx] += L2_seg @ (F_vals * u_seg)
            
            start_idx = end_idx
        
        # 加权累加
        total_contribution += weight * stage_contribution
    
    # 最终更新: u^{n+1} = u^n + Δt * total_contribution
    u_new = u_n + dt * total_contribution
    
    return u_new
```

### 10.4 solver.py中的时间循环逻辑

```python
def solve_time_evolution():
    """时间演化求解主函数"""
    
    # 初始化: 直接使用解值
    u_current = initialize_solution()  # 使用神经网络预测或随机初始化
    
    # 预编译所有算子
    fitter.fitter_init(model)
    
    # 时间循环: 直接操作解值
    while T < time_final:
        # 自适应时间步
        if it == 0:
            dt = dt_config / 10  # 首步使用小时间步
        else:
            dt = dt_config
            
        # 稳定性约束
        if hasattr(fitter, "estimate_stable_dt"):
            dt_stable = fitter.estimate_stable_dt(u_current)
            dt = min(dt, dt_stable)
        
        # IMEX-RK时间步进 - 直接解值操作
        u_current = fitter.solve_time_step(u_current, dt)
        
        T += dt
        it += 1
    
    # 仅在最终输出时转换为系数格式
    coeffs_final = solution_to_coefficients(u_current)
    u_seg_list = reconstruct_segmented_solution(u_current)
    
    return u_current, u_seg_list, model, coeffs_final
```

### 10.5 设计优势分析

#### 10.5.1 计算效率优势
- **减少转换开销**：时间循环内避免频繁的系数-解值转换
- **预编译加速**：算子矩阵一次编译，多次重用
- **内存优化**：解值直接操作减少中间变量存储

#### 10.5.2 数值稳定性优势  
- **减少累积误差**：避免多次转换带来的数值误差累积
- **算子一致性**：所有算子使用统一的解值接口
- **物理直觉**：解值操作更符合PDE求解的物理含义

#### 10.5.3 框架兼容性优势
- **向后兼容**：最终输出仍为系数格式，保持接口兼容
- **模块化设计**：solver和fitter职责清晰分离
- **扩展性良好**：易于扩展到其他时间积分格式

### 10.6 与传统方法的对比

| 特性 | 传统系数方法 | DeePoly解值方法 |
|------|-------------|----------------|
| 时间循环操作对象 | 系数向量 β | 解值向量 u |
| 算子调用方式 | L(Vβ), N(Vβ) | L(u), N(u) |
| 转换频率 | 每时间步多次 | 仅最终输出 |
| 数值稳定性 | 累积转换误差 | 直接操作稳定 |
| 计算效率 | 转换开销大 | 预编译高效 |
| 物理直觉性 | 抽象系数 | 直观解值 |

### 10.7 实现要点总结

1. **预编译策略**：分层预编译所有算子，一次计算多次使用
2. **解值操作**：时间循环直接操作解值，避免系数转换
3. **算子统一**：N和F算子直接接受u值输入，保持接口一致
4. **最终转换**：仅在输出时进行解值到系数转换
5. **稳定性控制**：自适应时间步和稳定性估计确保数值稳定

## 11. 参考文献

1. Ascher, U. M., Ruuth, S. J., & Spiteri, R. J. (1997). Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations.
2. Pareschi, L., & Russo, G. (2005). Implicit-explicit Runge-Kutta schemes and applications to hyperbolic systems with relaxation.
3. Boscarino, S., Pareschi, L., & Russo, G. (2013). Implicit-explicit Runge-Kutta schemes for hyperbolic systems and kinetic equations in the diffusion limit.