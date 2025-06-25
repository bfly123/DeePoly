# IMEX Runge-Kutta Time Integration Schemes

## 1. Introduction

本文档介绍了用于求解时间相关偏微分方程的隐式-显式（Implicit-Explicit, IMEX）龙格-库塔方法。IMEX方法特别适用于包含不同刚性特征的算子分离问题。

## 2. General Problem Formulation

### 2.1 Single Equation Form

$$\frac{\partial u}{\partial t} = L_1(u) + N(u) + L_2(u) f(u)$$

### 2.2 System of Equations Form

对于包含 $m$ 个变量 $\mathbf{u} = [u_1, u_2, \ldots, u_m]^T$ 的方程组系统：

$$\frac{\partial \mathbf{u}}{\partial t} = \mathbf{L_1}(\mathbf{u}) + \mathbf{N}(\mathbf{u}) + \mathbf{L_2}(\mathbf{u}) \mathbf{f}(\mathbf{u})$$

### 2.3 Operator Classification

- **$\mathbf{L_1}(\mathbf{u})$**: 主线性算子 (隐式处理)
  - 通常包含扩散项、高阶导数项
  - 需要隐式处理以保证数值稳定性
  
- **$\mathbf{N}(\mathbf{u})$**: 完全非线性项 (显式处理)  
  - 通常包含源项等
  - 可以显式处理以提高计算效率
  
- **$\mathbf{L_2}(\mathbf{u}) \mathbf{f}(\mathbf{u})$**: 半隐式非线性项
  - $\mathbf{L_2}(\mathbf{u})$: 线性算子 (隐式处理)
  - $\mathbf{f}(\mathbf{u})$: 非线性函数 (显式处理)
  - 例如非线性对流项，可以将波速看做$\mathbf{f}$
  - 适用于具有线性主导部分的非线性项

## 3. General IMEX-RK Framework

### 3.1 Butcher Tableau Structure

对于 $s$-阶段IMEX-RK方法，定义两套Butcher表：

**隐式表 (Implicit Tableau):**
$$\tilde{A} = (\tilde{a}_{ij})_{s \times s}, \quad \tilde{b} = (\tilde{b}_i)_{s \times 1}, \quad \tilde{c} = (\tilde{c}_i)_{s \times 1}$$

**显式表 (Explicit Tableau):**
$$A = (a_{ij})_{s \times s}, \quad b = (b_i)_{s \times 1}, \quad c = (c_i)_{s \times 1}$$

通常满足：$c = \tilde{c}$ 和 $b = \tilde{b}$

### 3.2 Stage Equations (General Form)

对于第 $i$ 个阶段 ($i = 1, 2, \ldots, s$)，阶段解 $\mathbf{U}^{(i)}$ 满足：

$$\mathbf{U}^{(i)} = \mathbf{u}^n + \Delta t \sum_{j=1}^{i-1} a_{ij} \mathbf{N}(\mathbf{U}^{(j)}) + \Delta t \sum_{j=1}^{s} \tilde{a}_{ij} [\mathbf{L_1}(\mathbf{U}^{(j)}) + \mathbf{L_2}(\mathbf{U}^{(j)}) \mathbf{f}(\mathbf{U}^{(j)})]$$

### 3.3 Linear System Construction

重新整理阶段方程，将所有包含 $\mathbf{U}^{(i)}$ 的项移到左边：

$$\left[\mathbf{I} - \Delta t \tilde{a}_{ii} \mathbf{L_1} - \Delta t \tilde{a}_{ii} \mathbf{L_2} \text{diag}(\mathbf{f}(\mathbf{U}^{prev}))\right] \mathbf{U}^{(i)} = \mathbf{RHS}^{(i)}$$

其中右端项为：
$$\mathbf{RHS}^{(i)} = \mathbf{u}^n + \Delta t \sum_{j=1}^{i-1} [a_{ij} \mathbf{N}(\mathbf{U}^{(j)}) + \tilde{a}_{ij} (\mathbf{L_1}(\mathbf{U}^{(j)}) + \mathbf{L_2}(\mathbf{U}^{(j)}) \mathbf{f}(\mathbf{U}^{(j)}))]$$

### 3.4 Final Time Step Update

$$\mathbf{u}^{n+1} = \mathbf{u}^n + \Delta t \sum_{i=1}^{s} b_i [\mathbf{N}(\mathbf{U}^{(i)}) + \mathbf{L_1}(\mathbf{U}^{(i)}) + \mathbf{L_2}(\mathbf{U}^{(i)}) \mathbf{f}(\mathbf{U}^{(i)})]$$

## 4. IMEX-RK(2,2,2) Scheme

### 4.1 Butcher Tableau

**参数:**
$$\gamma = \frac{2-\sqrt{2}}{2} \approx 0.2928932$$

**隐式表:**
$$\tilde{A} = \begin{pmatrix} \gamma & 0 \\ 1-2\gamma & \gamma \end{pmatrix}, \quad \tilde{b} = \begin{pmatrix} \frac{1}{2} \\ \frac{1}{2} \end{pmatrix}, \quad \tilde{c} = \begin{pmatrix} \gamma \\ 1 \end{pmatrix}$$

**显式表:**
$$A = \begin{pmatrix} 0 & 0 \\ 1-\gamma & 0 \end{pmatrix}, \quad b = \begin{pmatrix} \frac{1}{2} \\ \frac{1}{2} \end{pmatrix}, \quad c = \begin{pmatrix} \gamma \\ 1 \end{pmatrix}$$

**标准Butcher表表示:**
$$
\begin{array}{c|cc}
\gamma & \gamma & 0 \\
1 & 1-2\gamma & \gamma \\
\hline
 & \frac{1}{2} & \frac{1}{2}
\end{array}
$$

### 4.2 Stage Equations

**阶段1:**
$$\left[\mathbf{I} - \gamma \Delta t \mathbf{L_1} - \gamma \Delta t \mathbf{L_2} \text{diag}(\mathbf{f}(\mathbf{u}^n))\right] \mathbf{U}^{(1)} = \mathbf{u}^n$$

**阶段2:**
$$\left[\mathbf{I} - \gamma \Delta t \mathbf{L_1} - \gamma \Delta t \mathbf{L_2} \text{diag}(\mathbf{f}(\mathbf{U}^{(1)}))\right] \mathbf{U}^{(2)} = \mathbf{u}^n + \Delta t(1-2\gamma) [\mathbf{L_1}(\mathbf{U}^{(1)}) + \mathbf{L_2}(\mathbf{U}^{(1)}) \mathbf{f}(\mathbf{U}^{(1)})] + \Delta t(1-\gamma) \mathbf{N}(\mathbf{U}^{(1)})$$

**最终更新:**
$$\mathbf{u}^{n+1} = \mathbf{u}^n + \frac{\Delta t}{2}[\mathbf{L_1}(\mathbf{U}^{(1)}) + \mathbf{L_1}(\mathbf{U}^{(2)}) + \mathbf{L_2}(\mathbf{U}^{(1)})\mathbf{f}(\mathbf{U}^{(1)}) + \mathbf{L_2}(\mathbf{U}^{(2)})\mathbf{f}(\mathbf{U}^{(2)}) + \mathbf{N}(\mathbf{U}^{(1)}) + \mathbf{N}(\mathbf{U}^{(2)})]$$

## 5. Application Examples

### 5.1 Allen-Cahn Equation
$$\frac{\partial u}{\partial t} = \epsilon^2 \nabla^2 u - u(u^2 - 1)$$

**算子分离:**
- $L_1(u) = \epsilon^2 \nabla^2 u$
- $L_2(u) = u$, $f(u) = -(u^2 - 1)$
- $N(u) = 0$

**阶段方程:**

*阶段1:*
$$\left(I - \gamma \Delta t \epsilon^2 \nabla^2 + \gamma \Delta t \text{diag}((u^n)^2 - 1)\right) U_1 = u^n$$

*阶段2:*
$$\left(I - \gamma \Delta t \epsilon^2 \nabla^2 + \gamma \Delta t \text{diag}((U_1)^2 - 1)\right) U_2 = u^n + \Delta t(1-2\gamma) [\epsilon^2 \nabla^2 U_1 - U_1 \odot ((U_1)^2 - 1)]$$

### 5.2 Cahn-Hilliard Equation
$$\frac{\partial u}{\partial t} = -\nabla^2(\epsilon^2 \nabla^2 u - u^3 + u)$$

**重写为:** $u_t = -\epsilon^2 \nabla^4 u - \nabla^2 u + \nabla^2 u \cdot u^2$

**算子分离:**
- $L_1(u) = -\epsilon^2 \nabla^4 u - \nabla^2 u$
- $L_2(u) = \nabla^2 u$, $f(u) = u^2$
- $N(u) = 0$

### 5.3 Convection-Diffusion-Reaction
$$\frac{\partial u}{\partial t} = \epsilon \nabla^2 u + \nabla \cdot (v u) - \alpha u^3$$

**算子分离:**
- $L_1(u) = \epsilon \nabla^2 u$
- $L_2(u) = \nabla \cdot (v u)$, $f(u) = 1$
- $N(u) = -\alpha u^3$

### 5.4 Reaction-Diffusion System
$$\begin{align}
\frac{\partial u}{\partial t} &= D_u \nabla^2 u + f(u,v) \\
\frac{\partial v}{\partial t} &= D_v \nabla^2 v + g(u,v)
\end{align}$$

**算子分离:**
- $\mathbf{L_1}(\mathbf{u}) = \begin{pmatrix} D_u \nabla^2 u \\ D_v \nabla^2 v \end{pmatrix}$
- $\mathbf{N}(\mathbf{u}) = \begin{pmatrix} f(u,v) \\ g(u,v) \end{pmatrix}$
- $\mathbf{L_2} = \mathbf{0}$, $\mathbf{f} = \mathbf{0}$

## 6. Implementation Considerations

### 6.1 Matrix Assembly

对于每个阶段，需要构建并求解线性系统：
$$\mathbf{A}^{(i)} \mathbf{U}^{(i)} = \mathbf{b}^{(i)}$$

其中系统矩阵为：
$$\mathbf{A}^{(i)} = \mathbf{I} - \gamma \Delta t \mathbf{L_1} - \gamma \Delta t \mathbf{L_2} \text{diag}(\mathbf{f}(\mathbf{U}^{prev}))$$

### 6.2 Computational Efficiency

**DIRK特性:** 对于IMEX-RK(2,2,2)，两个阶段的系统矩阵具有相同的结构（对角元素都是$\gamma$），可以重复使用矩阵分解。

**稀疏性:** 利用偏微分方程离散化矩阵的稀疏特性，使用稀疏矩阵求解器。

### 6.3 Stability Analysis

**稳定性约束:**
$$\Delta t \leq \frac{C_{stability}}{\max(\rho(\mathbf{L_1}), \rho(\mathbf{L_2}) \max|\mathbf{f}'(\mathbf{u})|, \max|\mathbf{N}'(\mathbf{u})|)}$$

其中：
- $\rho(\cdot)$ 表示矩阵的谱半径
- $C_{stability} \approx 1.5$ 对于IMEX-RK(2,2,2)

## 7. Other IMEX-RK Schemes

### 7.1 IMEX-RK(1,1,1) - Backward Euler/Forward Euler
$$A = (0), \quad \tilde{A} = (1), \quad b = \tilde{b} = (1)$$

### 7.2 IMEX-RK(2,3,2) - ARS(2,3,2)
$$\gamma = \frac{3 + \sqrt{3}}{6}$$

### 7.3 IMEX-RK(3,4,3) - ARS(3,4,3)
高阶格式，适用于需要高精度的问题。

## 8. Algorithm Implementation Framework

```python
def imex_rk_step(u_n, dt, A_exp, A_imp, b, c, L1, L2, N, f):
    """
    Generic IMEX-RK time step
    
    Parameters:
    - u_n: Solution at time t_n
    - dt: Time step size
    - A_exp, A_imp: Explicit and implicit Butcher tableau matrices
    - b, c: Butcher tableau vectors
    - L1, L2, N, f: Operators and functions
    """
    s = len(b)  # Number of stages
    U = [None] * s  # Stage solutions
    
    for i in range(s):
        # Build RHS
        rhs = u_n.copy()
        for j in range(i):
            rhs += dt * A_exp[i,j] * N(U[j])
            rhs += dt * A_imp[i,j] * (L1(U[j]) + L2(U[j]) @ f(U[j]))
        
        if A_imp[i,i] == 0:  # Explicit stage
            U[i] = rhs
        else:  # Implicit stage
            # Build system matrix
            f_prev = f(U[i-1]) if i > 0 else f(u_n)
            A_matrix = I - dt * A_imp[i,i] * L1 - dt * A_imp[i,i] * L2 @ diag(f_prev)
            U[i] = solve_linear_system(A_matrix, rhs)
    
    # Final update
    u_new = u_n.copy()
    for i in range(s):
        u_new += dt * b[i] * (N(U[i]) + L1(U[i]) + L2(U[i]) @ f(U[i]))
    
    return u_new
```

## 9. References

1. Ascher, U. M., Ruuth, S. J., & Spiteri, R. J. (1997). Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations.
2. Pareschi, L., & Russo, G. (2005). Implicit-explicit Runge-Kutta schemes and applications to hyperbolic systems with relaxation.
3. Boscarino, S., Pareschi, L., & Russo, G. (2013). Implicit-explicit Runge-Kutta schemes for hyperbolic systems and kinetic equations in the diffusion limit.

# 程序实现方案

先给出整体方程（去掉时间项）
```json

   eq:{"L_1(u) + N(u)+L_2(u)*F(u)"}，
   eq_L1:{"L_1(u)"},
   eq_N:{"N(u)"},
   "var":{}
    "var
    "constants:
```
config.py 中 解析整体eq 导数信息列表，根据导数列表生成L_1和L_2的导数列表和N和F的导数列表

在fitter.py中构造 L_1(U)和 L_2(U)算子矩阵，根据线性参数生成函数 L_1（U*beta）和L_2(U*beta)，以及函数 N（beta）和 F（beta）
基于分区组合。

基于时间格式调用这些函数组成时间推进。

1，通过任意线性和非线性算子形成 $F(u)$函数
2,通过任意线性算子形成 $L(U)$函数  其中u =\sum U\cdot \beta$ $U$是基函数。所以 $L(U)$是固定的，初始就可以生成。后续只需要更新 $\beta$
根据config算子读入，建立算子函数