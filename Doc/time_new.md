# Second-Order Time Integration Schemes

## 1. Purely Explicit 2nd Order (Heun’s Method / Explicit RK2)

For \( \frac{du}{dt} = f(u, t) \):

\[
\begin{align*}
k_1 &= f(u^n, t^n) \\
k_2 &= f(u^n + \Delta t\, k_1, t^n + \Delta t) \\
u^{n+1} &= u^n + \frac{\Delta t}{2}(k_1 + k_2)
\end{align*}
\]

转换为格式形式：
\[
  \begin{align*}
  u^1 - u^n = dt f(u^n)\\
  u^2 - u^n = dt \Delta f(u^1 ) 

  u^{n+1} = u^n  + 0.5 \Delta t (f(u^1) + f(u^2))
  还是 
  u^{n+1} = u^n  + 0.5 \Delta t (f(u^n) + f(u^2))


]

---

## 2. Purely Implicit 2nd Order (Crank-Nicolson)

For \( \frac{du}{dt} = f(u, t) \):

\[
u^{n+1} = u^n + \frac{\Delta t}{2} \left[ f(u^n, t^n) + f(u^{n+1}, t^{n+1}) \right]
\]

---

## 3. IMEX-SSP2 (Second-Order IMEX Runge-Kutta)

For \( \frac{du}{dt} = F(u, t) + G(u, t) \), where \( F \) is explicit, \( G \) is implicit.

A common IMEX-SSP2 scheme (ARS(2,2,2)):

\[
\begin{align*}
U^{(1)} &= u^n + \gamma \Delta t\, G(U^{(1)}, t^n + \gamma \Delta t) \\
U^{(2)} &= u^n + \Delta t\, F(U^{(1)}, t^n + \gamma \Delta t) + (1-2\gamma)\Delta t\, G(U^{(1)}, t^n + \gamma \Delta t) + \gamma \Delta t\, G(U^{(2)}, t^n + \Delta t) \\
u^{n+1} &= U^{(2)}
\end{align*}
\]

where \( \gamma = 1 - \frac{1}{\sqrt{2}} \approx 0.292893 \).

---

### Summary Table

| Scheme Type      | Formula                                                                                  | Notes                |
|------------------|-----------------------------------------------------------------------------------------|----------------------|
| Explicit RK2     | \( u^{n+1} = u^n + \frac{\Delta t}{2}(k_1 + k_2) \)                                     | All explicit         |
| Implicit CN      | \( u^{n+1} = u^n + \frac{\Delta t}{2}[f(u^n) + f(u^{n+1})] \)                           | All implicit         |
| IMEX-SSP2        | See above (ARS(2,2,2) or IMEX SSP2)                                                     | Mixed explicit/implicit |
