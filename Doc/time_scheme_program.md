
# IMEX-RK(2,2,2) 时间积分格式实现文档

## 通用IMEX-RK公式



求解ODE系统：$\frac{d\mathbf{U}}{dt} = \mathbf{A}(\mathbf{U})$（通用形式）

### 纯显式SSP2格式

**Butcher表（显式部分）：**
```
c | A
--|------
0 | 0   0
1 | 1   0
--|------
  |1/2 1/2
```

**时间推进公式：** 针对方程 $\frac{d\mathbf{U}}{dt} = \mathbf{A}(\mathbf{U})$

**阶段1：** 
$$\mathbf{U}^{(1)} = \mathbf{U}^n + \Delta t \cdot \mathbf{A}(\mathbf{U}^n)$$

**阶段2：** 
$$\mathbf{U}^{(2)} = \mathbf{U}^n + \Delta t \cdot \mathbf{A}(\mathbf{U}^{(1)})$$

**最终更新：**
$$\mathbf{U}^{n+1} = \frac{1}{2}\mathbf{U}^n + \frac{1}{2}\mathbf{U}^{(2)}$$

**等价形式：**
$$\mathbf{U}^{n+1} = \mathbf{U}^n + \frac{\Delta t}{2} \mathbf{A}(\mathbf{U}^n) + \frac{\Delta t}{2} \mathbf{A}(\mathbf{U}^{(1)})$$

**稳定性：** CFL条件 $\Delta t \leq \frac{1}{|\lambda_{max}|}$（强稳定保持性质）

### 纯隐式SSP2格式

**Butcher表（隐式部分）：**
```
c   | A
----|------
γ   | γ   0
1-γ | 1-2γ γ
----|------
    |1/2 1/2
```

**时间推进公式：** 针对方程 $\frac{d\mathbf{U}}{dt} = \mathbf{B}(\mathbf{U})$

**阶段1：** 
$$\mathbf{U}^{(1)} = \mathbf{U}^n + \gamma \Delta t \cdot \mathbf{B}(\mathbf{U}^{(1)})$$

**阶段2：** 
$$\mathbf{U}^{(2)} = \mathbf{U}^n + (1-2\gamma) \Delta t \cdot \mathbf{B}(\mathbf{U}^{(1)}) + \gamma \Delta t \cdot \mathbf{B}(\mathbf{U}^{(2)})$$

**最终更新：**
$$\mathbf{U}^{n+1} = \mathbf{U}^n + \frac{\Delta t}{2} \left[\mathbf{B}(\mathbf{U}^{(1)}) + \mathbf{B}(\mathbf{U}^{(2)})\right]$$

**参数：** $\gamma = 1 - \frac{1}{\sqrt{2}} \approx 0.2929$

**稳定性：** L-stable（无条件稳定，适合刚性问题）


**时间推进公式：** 针对方程 $\frac{d\mathbf{U}}{dt} = \mathbf{A}(\mathbf{U}) \mathbf{B}(\mathbf{U})$

**阶段1：**
$$\mathbf{U}^{(1)} = \mathbf{U}^n + \gamma \Delta t \cdot \mathbf{B}(\mathbf{U}^{(1)}) \cdot \mathbf{A}(\mathbf{U}^n)$$

**阶段2：**
$$\mathbf{U}^{(2)} = \mathbf{U}^n + (1-2\gamma) \Delta t \cdot \mathbf{B}(\mathbf{U}^{(1)}) \cdot \mathbf{A}(\mathbf{U}^n) + \gamma \Delta t \cdot \mathbf{B}(\mathbf{U}^{(2)}) \cdot \mathbf{A}(\mathbf{U}^{(1)})$$

**最终更新：**
$$\mathbf{U}^{n+1} = \mathbf{U}^n + \frac{\Delta t}{2} \left[\mathbf{B}(\mathbf{U}^{(1)}) \cdot \mathbf{A}(\mathbf{U}^n) + \mathbf{B}(\mathbf{U}^{(2)}) \cdot \mathbf{A}(\mathbf{U}^{(2)})\right]$$

**参数：** $\gamma = 1 - \frac{1}{\sqrt{2}} \approx 0.2929$

**特点：**
- 这是一个IMEX格式，其中$\mathbf{B}(\mathbf{U})$采用隐式处理，$\mathbf{A}(\mathbf{U})$采用显式处理
- 保持了二阶精度
- 适合处理刚性问题




### IMEX-RK(2,2,2)格式

**通用IMEX-RK格式：**
$$\mathbf{U}^{(k)} = \mathbf{U}^n - \Delta t \sum_{l=1}^{k-1} \tilde{a}_{kl} \mathbf{F}(t^n + \tilde{c}_l \Delta t, \mathbf{U}^{(l)}) + \Delta t \sum_{l=1}^{\rho} a_{kl} \mathbf{S}(t^n + c_l \Delta t, \mathbf{U}^{(l)})$$

$$\mathbf{U}^{n+1} = \mathbf{U}^n - \Delta t \sum_{k=1}^{\rho} \tilde{\omega}_k \mathbf{F}(t^n + \tilde{c}_k \Delta t, \mathbf{U}^{(k)}) + \Delta t \sum_{k=1}^{\rho} \omega_k \mathbf{S}(t^n + c_k \Delta t, \mathbf{U}^{(k)})$$




### 乘法系统分析：$\frac{d\mathbf{U}}{dt} = \mathbf{A}(\mathbf{U}) \cdot \mathbf{B}(\mathbf{U})$

对于系统 $\frac{d\mathbf{U}}{dt} = \mathbf{A}(\mathbf{U}) \cdot \mathbf{B}(\mathbf{U})$，其中：
- $\mathbf{A}(\mathbf{U})$ 需要**显式处理**
- $\mathbf{B}(\mathbf{U})$ 需要**隐式处理**

**可能的形式：**
1. **标量相乘**：$\frac{d\mathbf{U}}{dt} = A(\mathbf{U}) \cdot B(\mathbf{U})$
2. **矩阵-向量**：$\frac{d\mathbf{U}}{dt} = \mathbf{A}(\mathbf{U}) \cdot \mathbf{B}(\mathbf{U})$

**IMEX分离策略：**

**策略1 - 直接分离：**
$$\frac{d\mathbf{U}}{dt} = \overbrace{\mathbf{A}(\mathbf{U})}^{\mathbf{F}(\mathbf{U})} \cdot \overbrace{\mathbf{B}(\mathbf{U})}^{\mathbf{S}(\mathbf{U})}$$

**策略2 - 线性化分离：**
$$\frac{d\mathbf{U}}{dt} = \mathbf{A}(\mathbf{U}^n) \cdot \mathbf{B}(\mathbf{U}) + \mathbf{A}(\mathbf{U}) \cdot \mathbf{B}(\mathbf{U}^n) - \mathbf{A}(\mathbf{U}^n) \cdot \mathbf{B}(\mathbf{U}^n)$$

**策略3 - 近似分离：**
$$\frac{d\mathbf{U}}{dt} \approx \mathbf{A}(\mathbf{U}^n) \cdot \mathbf{B}(\mathbf{U}) + [\mathbf{A}(\mathbf{U}) - \mathbf{A}(\mathbf{U}^n)] \cdot \mathbf{B}(\mathbf{U}^n)$$

**应用IMEX-RK(2,2,2)：**

以策略1为例：
**阶段1：**
$$\mathbf{U}^{(1)} = \mathbf{U}^n + \gamma \Delta t \cdot \mathbf{A}(\mathbf{U}^n) \cdot \mathbf{B}(\mathbf{U}^{(1)})$$

**阶段2：**
$$\mathbf{U}^{(2)} = \mathbf{U}^n + (1-2\gamma) \Delta t \cdot \mathbf{A}(\mathbf{U}^{(1)}) \cdot \mathbf{B}(\mathbf{U}^{(1)}) + \gamma \Delta t \cdot \mathbf{A}(\mathbf{U}^{(1)}) \cdot \mathbf{B}(\mathbf{U}^{(2)})$$

**最终更新：**
$$\mathbf{U}^{n+1} = \mathbf{U}^n + \frac{\Delta t}{2} \left[ \mathbf{A}(\mathbf{U}^{(1)}) \cdot \mathbf{B}(\mathbf{U}^{(1)}) + \mathbf{A}(\mathbf{U}^{(2)}) \cdot \mathbf{B}(\mathbf{U}^{(2)}) \right]$$ 

for

$$\frac{\partial{U}}{\partial t} = L_1(U) + L_2(U)F(U)+ N(U)$$



其中 $L_1(U)$和 $L_2(U)$采用隐式二阶，而 F(U)和N(U)采用显式，基于上述公式

第一步：
$$U^1 = U^n + dt [L_1(U^1) + L_2(U^1)F(U^n) + N(U^n)] $$

$$U^2 = U^n + dt\frac{[L(U^{n+1}) + L(U^n)]}{2} + dt \frac{L_2(U^{n+1})+L_2(U^n)}{2} \frac{F(U^1)+F(U^n)}{2} + \frac{N(U^1)+N(U^n)}{2} $$
