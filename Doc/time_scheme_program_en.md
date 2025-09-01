# IMEX-RK(2,2,2) Time Integration Scheme

## Mathematical Formulation

### Stage 1
Solve for coefficients $\boldsymbol{\beta}^{(1)}$ implicitly:

$$\left[\mathbf{V} - \gamma \Delta t \mathbf{L}_1 - \gamma \Delta t \mathbf{L}_2 \odot \mathbf{F}(\mathbf{U}^n)\right] \boldsymbol{\beta}^{(1)} = \mathbf{U}^n + \gamma \Delta t \mathbf{N}(\mathbf{U}^n)$$

Compute stage solution:
$$\mathbf{U}^{(1)} = \mathbf{V} \boldsymbol{\beta}^{(1)}$$

### Stage 2
Solve for coefficients $\boldsymbol{\beta}^{(2)}$ implicitly:

$$\begin{aligned}
&\left[\mathbf{V} - \gamma \Delta t \mathbf{L}_1 - \gamma \Delta t \mathbf{L}_2 \odot \mathbf{F}(\mathbf{U}^{(1)})\right] \boldsymbol{\beta}^{(2)} \\
&= \mathbf{U}^n + \Delta t(1-2\gamma) \left[\mathbf{L}_1 + \mathbf{L}_2 \odot \mathbf{F}(\mathbf{U}^{(1)}) \right] \boldsymbol{\beta}^{(1)} \\
&\quad + \Delta t(1-\gamma) \mathbf{N}(\mathbf{U}^{(1)})
\end{aligned}$$

Compute stage solution:
$$\mathbf{U}^{(2)} = \mathbf{V} \boldsymbol{\beta}^{(2)}$$

### Final Time Step Update

$$
\mathbf{U}^{n+1} = \mathbf{U}^n + \Delta t \sum_{i=1}^{2} b_i \left[
\mathbf{L}_1 \boldsymbol{\beta}^{(i)} + \mathbf{N}(\mathbf{U}^{(i)}) + \mathbf{L}_2 \boldsymbol{\beta}^{(i)} \odot \mathbf{F}(\mathbf{U}^{(i)})
\right]
$$

where $b_1 = b_2 = 0.5$ and $\gamma = \frac{2-\sqrt{2}}{2} \approx 0.2928932$

## Butcher Tableaux

### Implicit Part (A_implicit)
$$
\begin{array}{c|cc}
\gamma & \gamma & 0 \\
1-\gamma & 1-2\gamma & \gamma \\
\hline
& 0.5 & 0.5
\end{array}
$$

### Explicit Part (A_explicit)  
$$
\begin{array}{c|cc}
0 & 0 & 0 \\
1-\gamma & 1-\gamma & 0 \\
\hline
& 0.5 & 0.5
\end{array}
$$

## Key Implementation Notes

1. **Operator Characteristics**:
   - L1, L2 operators have same dimensions as feature matrix V
   - F, N operators output same dimensions as solution U
   - Coefficients grouped by equation: `[n_eqs, dgN]`

2. **L2⊙F Term Handling**:
   The term $\mathbf{L}_2 \boldsymbol{\beta}^{(i)} \odot \mathbf{F}(\mathbf{U}^{(i)})$ is implemented as:
   ```
   L2_beta = L2_operators @ coeffs  # (n_points,)
   F_vals = F_func(features, U_stage)  # (n_points, n_eqs)
   L2F_contrib = L2_beta * F_vals    # element-wise multiplication
   ```

3. **Jacobian Matrix**:
   $$\mathbf{J} = \mathbf{V} - \gamma \Delta t \mathbf{L}_1 - \gamma \Delta t \text{diag}(\mathbf{F}) \mathbf{L}_2$$

4. **Key Corrections**:
   - Time step formula uses weights $b_i$, not fixed $\frac{1}{2}$
   - L2⊙F term order: $\mathbf{L}_2 \boldsymbol{\beta} \odot \mathbf{F}$, not $\mathbf{L}_2 \odot \mathbf{F} \boldsymbol{\beta}$