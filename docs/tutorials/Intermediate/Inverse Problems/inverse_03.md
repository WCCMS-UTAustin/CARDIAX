# Inverse Problems: Code Walkthrough

The full example is at `tutorials/Intermediate/inverse_model/example.py`. This page walks through each block in order.

## Imports

```python
import jax
import jax.numpy as np
import numpy as onp
from pathlib import Path
import os

from cardiax import box_mesh
from cardiax import FiniteElement, Problem, Newton_Solver
```

`jax.numpy` is used for all computation inside traced regions. `numpy` (aliased as `onp`) is used for postprocessing and PyVista calls, which require host arrays.

## Problem Class

```python
class HyperElasticity(Problem):
    def get_tensor_map(self):
        def psi(F, E):
            nu = 0.45
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.) * (Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2.
            return energy[0]

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, E):
            I = np.eye(3)
            F = u_grad + I
            P = P_fn(F, E)
            return P

        return first_PK_stress
```

`get_tensor_map` returns the pointwise stress function `first_PK_stress(u_grad, E) -> P`. The argument `E` is a scalar — the quadrature-point value of the stiffness field at a single integration point. `jax.grad(psi)` differentiates $\Psi$ with respect to its first argument ($\mathbf{F}$) only; $E$ is treated as a parameter. The `[0]` index on `energy` unwraps the scalar from a length-1 array, which is required by `jax.grad`.

```python
    def get_surface_maps(self):
        def right_trac(u, u_grad, x):
            return np.array([-10., 0., 0.])
        return {"u": {"right": right_trac}}
```

The traction is constant and does not depend on the displacement or position. The key `"right"` must match the key used in `location_fns` when constructing the problem.

```python
    def set_params(self, E):
        self.set_internal_vars({"u": {"E": self.fes["u"].convert_dof_to_quad(E)}})
```

`set_params` accepts `E` of shape `(n_nodes, 1)` and maps it to quadrature-point values via `convert_dof_to_quad`. The result — shape `(n_cells, n_quads, 1)` — is stored in `internal_vars` and passed to `first_PK_stress` at assembly time as the second argument.

## Mesh and FE Field

```python
Lx, Ly, Lz = 1., 1., 1.
mesh = box_mesh(Nx=10, Ny=10, Nz=10, Lx=Lx, Ly=Ly, Lz=Lz)
fe = FiniteElement(mesh, vec=3, dim=3, ele_type='hexahedron', gauss_order=1)
```

`vec=3` specifies a vector-valued displacement field in 3D. `gauss_order=1` places a single quadrature point per element, which is sufficient for trilinear hexahedra and keeps the per-element stiffness value scalar.

## Boundary Conditions

```python
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def zero_bc(point):
    return 0.

bc_left = [[left]*3, [0, 1, 2], [zero_bc]*3]
dirichlet_bc_info = {"u": [bc_left]}
location_fns = {"u": {"right": right}}
```

All three displacement components are pinned to zero on the left face via row elimination. The right face is registered as a Neumann surface; the key `"right"` links it to the traction returned by `get_surface_maps`.

## Forward Solve and Synthetic Data

```python
problem = HyperElasticity({"u": fe}, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)

E_true = 80
Es = E_true * np.ones((len(problem.mesh["u"].points), 1))
problem.set_params(Es)

solver = Newton_Solver(problem, np.zeros((problem.num_total_dofs_all_vars)))
sol_true, info = solver.solve(1e-8)
```

The forward problem is solved at $E^* = 80$ to generate synthetic displacement observations. `solver.solve(1e-8)` runs Newton's method to an absolute residual tolerance of $10^{-8}$ and returns `(sol, info)` where `info[0]` is a boolean convergence flag.

## Adjoint Wrapper and Loss

```python
fwd_pred = solver.ad_wrapper()

def error(sol):
    return np.linalg.norm(sol - sol_true)

def composed(E):
    sol = fwd_pred(E)
    return error(sol), sol

val_grad = jax.value_and_grad(composed, has_aux=True)
```

`ad_wrapper()` returns a differentiable callable `fwd_pred(E) -> u`. When `jax.value_and_grad` differentiates through `composed`, JAX hits the custom VJP boundary at `fwd_pred` and routes the gradient through `implicit_vjp` rather than through the Newton iteration.

`has_aux=True` designates the second return value of `composed` — the converged displacement array — as auxiliary data that passes through without contributing to the gradient. The return structure of `val_grad(E_guess)` is:

```
((loss, sol), grad_E)
```

## Inverse Solve Loop

```python
E_val = 50.
sol = np.zeros(problem.num_total_dofs_all_vars)
loss, tol = 1e3, 1e-2
sols, E_vals = [], [E_val]

while loss > tol:
    solver.initial_guess = sol
    E_guess = E_val * np.ones((len(problem.mesh["u"].points), 1))

    vals = val_grad(E_guess)
    loss = vals[0][0]
    sol  = vals[0][1]
    grad = vals[1].sum()

    E_val = E_val - 0.25 * loss / grad
    E_val = np.clip(E_val, 50, 100)

    sols.append(sol)
    E_vals.append(E_val)
```

A few points on each step:

- **`solver.initial_guess = sol`**: warm-starts the next Newton solve from the previous outer iterate. This must happen outside the traced region — inside `composed`, this write would be silently ignored by JAX.
- **`vals[1].sum()`**: `vals[1]` has shape `(n_nodes, 1)`, one gradient value per node. Summing to a scalar is valid here only because $E$ is spatially uniform. For spatially-varying estimation, pass the full gradient array to your optimizer.
- **`0.25 * loss / grad`**: a heuristic step size, not a line search. It is stable for this synthetic problem but is not general.
- **`np.clip`**: enforces a coarse feasibility bound in place of a regularizer. For a real inverse problem, use a positivity-preserving parameterization or a constrained optimizer.

## Postprocessing

```python
if plotting := True:
    import pyvista as pv

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = onp.array(sols[0]).reshape(-1, fe.dim)
    mesh.point_data["sol_true"] = onp.array(sol_true).reshape(-1, fe.dim)
    warped = mesh.warp_by_vector("sol", factor=1.)
    warped_true = mesh.warp_by_vector("sol_true", factor=1.)
    pl.add_mesh(warped)
    pl.add_mesh(warped_true, style='wireframe', color='black')
    pl.screenshot(fig_dir / "initial_guess.png")
    pl.close()
```

`onp.array(...)` transfers the JAX array to host memory before passing to PyVista, which does not accept JAX device arrays. The `.reshape(-1, fe.dim)` reshapes the flat DOF vector into a `(n_nodes, 3)` displacement array expected by `warp_by_vector`.

The GIF loop iterates over `sols`, warping the mesh by the estimated displacement at each outer iteration and overlaying the true solution as a wireframe.

---

*Previous: [Example Problem](inverse_02.md)*
