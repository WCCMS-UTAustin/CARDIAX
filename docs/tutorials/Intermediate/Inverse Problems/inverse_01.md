# Inverse Problems: CARDIAX Implementation

## Design Overview

The adjoint pattern described in the [previous section](inverse_00.md) is implemented in CARDIAX through three cooperating abstractions:

- **`Problem.set_params`**: the interface through which parameters enter the residual.
- **`Solver_Base.implicit_vjp`**: the analytic adjoint, called as the backward pass of the custom VJP.
- **`Solver_Base.ad_wrapper`**: registers the custom VJP with JAX and returns a differentiable forward prediction callable.

Together these give a callable `fwd_pred(p) -> u` that behaves like a standard JAX function — `jax.grad`, `jax.value_and_grad`, and `jax.jacobian` all work on it — while routing gradients through the analytic adjoint rather than through the Newton iteration.

## Parameter Interface: `set_params`

Every inverse-capable `Problem` subclass must implement:

```python
def set_params(self, params):
    ...
```

`set_params` is responsible for loading the parameter array into the problem's internal state so that the residual assembly picks it up. Whatever JAX-traceable operations are applied to `params` inside `set_params` become part of the computational graph that `implicit_vjp` differentiates through — so the shape and structure of the returned gradient corresponds directly to the shape of `params` as passed in. The specific implementation depends on how the parameter enters the constitutive law; see the [example](inverse_02.md) for a concrete case.

## Adjoint Computation: `implicit_vjp`

`implicit_vjp` implements Step 2 and Step 3 of the discrete adjoint algorithm. Its signature is:

```python
def implicit_vjp(self, sol, params, v, ...)
```

where `sol` is the converged primal $\mathbf{u}^*$, `params` is the parameter array $\mathbf{m}$, and `v` is the cotangent vector arriving from the objective (i.e., $(\partial J / \partial \mathbf{u})^T$).

Internally it:

1. Solves $\mathbf{K}^T \boldsymbol{\lambda} = \mathbf{v}$ by calling `jax_solve` with the row and column index arrays swapped, which transposes the sparse operator without forming $\mathbf{K}^T$ explicitly.
2. Evaluates $-\boldsymbol{\lambda}^T (\partial \mathbf{R}/\partial \mathbf{m})$ via `jax.vjp` applied to the constraint function `params -> R(u*, params)` with cotangent $\boldsymbol{\lambda}$.

The sign convention matches the derivation in the previous section: `implicit_vjp` returns the negated VJP result, i.e., $\mathrm{d}J/\mathrm{d}\mathbf{m}$.

## Custom VJP: `ad_wrapper`

`ad_wrapper` returns a Python callable registered with `jax.custom_vjp`:

```python
fwd_pred = solver.ad_wrapper()
```

The custom VJP has two components:

- **Forward function**: calls `solver.solve`, returns the converged DOF vector, and saves `(sol, params)` as residuals for the backward pass.
- **Backward function**: receives the cotangent from the loss, calls `implicit_vjp(sol, params, cotangent)`, and returns the parameter-space gradient.

This establishes a hard boundary in the JAX computation graph. JAX tracing does not enter the Newton loop; adaptive iteration counts, convergence checks, and any Python-level state updates inside `solve` are invisible to the tracer. What JAX sees at the boundary is simply: a function that maps `params` to `u`, with a known VJP.

### JIT Compatibility

`fwd_pred` is **not end-to-end JIT-compilable** because the Newton solve involves Python-level control flow (iteration count, convergence check). The outer optimization loop — the calls to `val_grad`, the parameter update, and any logging — runs in eager mode. This is intentional and not a limitation of the adjoint implementation.

What *can* be JIT-compiled is the adjoint solve itself: `implicit_vjp` is a pure JAX function and will be traced and compiled on first call.

## Warm-Starting the Newton Solve

The Newton solve inside `fwd_pred` reads its initial guess from `solver.initial_guess`. Mutating this field between outer optimization iterations:

```python
solver.initial_guess = sol
```

reduces Newton iteration counts significantly once the parameter estimate is close to the true solution, since the primal state changes smoothly as $\mathbf{m}$ changes. This assignment must occur **outside** the traced region. If placed inside `composed` or any function passed to `jax.grad`, the write will be traced over and silently ignored.

## Constructing the Loss and Gradient

The standard pattern for the outer optimization loop is:

```python
def composed(params):
    sol = fwd_pred(params)
    loss = objective(sol)
    return loss, sol

val_grad = jax.value_and_grad(composed, has_aux=True)
```

`has_aux=True` tells JAX that `composed` returns `(primary, auxiliary)` and that only `primary` should be differentiated. The return of `val_grad(params)` is `((loss, sol), grad_m)`. This pattern lets the optimizer access both the converged displacement field and the gradient in a single adjoint pass.

---

*Previous: [Theory](inverse_00.md) | Next: [Example Problem](inverse_02.md)*
