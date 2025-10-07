# General PDE Solver 测试结果摘要

## 测试配置
- 案例: Allen-Cahn 稳态方程 1D
- Spotter训练轮数: 1000 epochs
- 伪时间步最大迭代: 500
- 收敛容差: 1e-4

## 测试结果

### Spotter阶段 ✓
- 训练完成: 1000 epochs
- 最终物理损失: 1.886005e-07
- 初始解范数: ||U|| = 8.628929e-03
- 最大值: max|U| = 1.984915e-04

### 伪时间步阶段 ❌ **严重问题**
- 迭代次数: 10次就"收敛"
- 最终残差: 0.000000e+00
- **解全是零**: ||U|| = 0.000000e+00

## 异常现象

1. **线性系统求解异常**:
   - ||b||_2 = 0.000000e+00 (右端项为零!)
   - ||Ax-b||_2 = 0.000000e+00
   - 所有迭代的残差都是0

2. **解退化为零**:
   - 从Spotter的非零解 (||U|| = 8.6e-3)
   - 变成伪时间步后的零解 (||U|| = 0.0)

3. **异常快速"收敛"**:
   - 仅10次迭代就达到零残差
   - 每次迭代的||b||都是0

## 可能原因

### 假设1: 初值传递失败
- Spotter得到的U_init没有正确传递给伪时间步
- 导致从零初值开始

### 假设2: RHS构建错误
- IMEX格式的右端项 b = U^n + dt*(S - N) 计算错误
- 可能 U^n, S, N 都是零

### 假设3: 源项数据流问题
- source_segments 未正确传递
- 导致 S_n 全为零

### 假设4: 算子计算问题
- N算子 (-5*u+5*u**3) 计算异常
- 当u=0时,N=0,S=0,导致RHS=U^n=0

## 需要检查的代码位置

1. `solver.py` 中的 U_init 传递:
   - `solve()` 方法中 `solve_steady_state(U_init=U_init)`

2. `imex_1st.py` 中的 RHS 计算:
   - 第133行: `rhs_eq += dt * (S_n[:, eq_idx] - N_n[eq_idx])`

3. `GeneralPDEDataGenerator` 中的源项:
   - `_load_source_term()` 方法
   - `source_segments` 字段

4. `GeneralPDEFitter` 中的初值设置:
   - `solve_steady_state()` 的 U_init 参数处理

## 下一步行动

1. 添加调试输出查看:
   - Spotter输出的U_init具体值
   - 伪时间步第1次迭代的U_n
   - IMEX中的S_n, N_n值

2. 检查U_init是否正确传递

3. 验证source_segments是否在data字典中

4. 检查solver.py中的_spotter_stage()是否返回U_init
