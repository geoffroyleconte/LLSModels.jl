# LLSModels.jl Tutorial

## Create a LLSModel:

We can define a linear least squares by passing the matrices that define the problem
```math
\begin{aligned}
\min \quad & \tfrac{1}{2}\|Ax - b\|^2 = \tfrac{1}{2}\|F(x)\|^2 \\
& c_L  \leq Cx \leq c_U \\
& \ell \leq  x \leq u.
\end{aligned}
```

```@example lls
using LLSModels
m, n = 2, 3
A = rand(m, n)
b = rand(m)
lcon, ucon = zeros(m), fill(Inf, m)
C = ones(m, n)
lvar, uvar = fill(-10.0, n), fill(200.0, n)
lls = LLSModel(A, b, lvar = lvar, uvar = uvar, C = C, lcon = lcon, ucon = ucon)
```

## Use the NLPModels.jl API

You can use the [`NLPModels.jl API for NLSModels`](https://juliasmoothoptimizers.github.io/NLPModels.jl/stable/api/#nls-api) to access the residual value and its derivatives:

```@example lls
using NLPModels
x1 = rand(n)
residual(lls, x1) # F(x)
```

```@example lls
jac_residual(lls, x1) # JF(x)
```

## Model interaction

You can convert a linear least-squares problem with residual `F(x)` to a nonlinear optimization problem with constraints `F(x) = r` and objective `¹/₂‖r‖²`.

```@example lls
using NLPModelsModifiers
FLLS = FeasibilityFormNLS(lls)
```

Then, it is possible to create a QuadraticModel from this problem.

```@example lls
using QuadraticModels
QM = QuadraticModel(FLLS, FLLS.meta.x0)
```

## Solving with RipQP

[`RipQP.jl`](https://github.com/JuliaSmoothOptimizers/RipQP.jl) is a regularized interior point solver that can solve LLSModels.
The optimal value `x_opt` and its associated residual `r_opt = A * x_opt - b` can be obtained with:

```@example lls
using RipQP
stats = ripqp(lls)
x_opt, r_opt = stats.solution[1:n], stats.solution[(n + 1):end]
```