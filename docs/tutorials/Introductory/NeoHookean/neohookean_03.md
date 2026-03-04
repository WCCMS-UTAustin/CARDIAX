
# JAX-Aware Implementation for Changing Boundary Conditions

A very critical component for maintaining high speed simulations within `CARDIAX` is knowing how to appropriately modify problems to keep high efficiency. While we have done our best to try and keep everything optimally performant in a vast number of scenarios, we obviously haven't caught them all. This example showcases one such scenario.

A common problem in biomechanics is solving many boundary value problems in series given experimental data. A prime example is when force-displacement data is measured for material testing. This tutorial is strictly showcasing the loss of performance when these dirichlet values are set inappropriately. Other tutorials will be linked once completed to show how to post-process simulations for desired quantities and how to solve inverse problems to find the material parameters.

## Implementation

We import our standard modules, but notice that we now import functools as well.

```python
import jax
import jax.numpy as np
import time
import functools as fctls

from cardiax import box_mesh
from cardiax import Problem
from cardiax import Newton_Solver
from cardiax import FiniteElement
```

We define our standard NeoHookean model with no surface integrals.

```python
class HyperElasticity(Problem):

    def get_tensor_map(self):

        def psi(F):
            E = 10.
            nu = 0.3
            mu = E / (2. * (1. + nu))
            kappa = E / (3. * (1. - 2. * nu))
            J = np.linalg.det(F)
            I1 = np.trace(F.T @ F)
            energy = (mu / 2.) * (I1 - 3. - 2 * np.log(J)) + (kappa / 2.) * (J - 1.)**2.
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(u_grad.shape[0])
            F = u_grad + I
            P = P_fn(F)
            return P

        return first_PK_stress
```

Now create the unit cube with appropriate field.

```python
# Specify mesh-related information (first-order hexahedron element).
mesh = box_mesh(Nx=10, Ny=10, Nz=10, Lx=1., Ly=1., Lz=1.)
fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = "hexahedron", gauss_order = 1)
```

We will apply boundary conditions to the left and right face.

```python
# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1., atol=1e-5)
```

We want to apply a simple tension test, so we need a `zero` function to fix the y and z coordinates while moving the x coordinate a desired amount.

```python
# Define Dirichlet boundary values.
def zero(point):
    return 0.

def val(value, point):
    return value
```

We now parameterize the setting of the dirichlet boundary conditions. For example, we set the initial values of the left and right face to move in the x direction by -0.1 and 0.1 respectively.

```python
def set_bcs(value):
    
    bc_left = [
        [left] * 3, 
        [0, 1, 2],
        [fctls.partial(val, -value), zero, zero]
        ]
    bc_right = [
        [right] * 3, 
        [0, 1, 2],
        [fctls.partial(val, value), zero, zero]
        ]
    return {"u": [bc_left, bc_right]}

dirichlet_bc_info = set_bcs(0.1)
```

We now initialize the problem with these boundary conditions and the solver.

```python
problem = HyperElasticity({"u": fe},
                          dirichlet_bc_info=dirichlet_bc_info)
solver = Newton_Solver(problem, np.zeros((problem.num_total_dofs_all_vars)))
```

We now time the initial solve where JIT compilation occurs for comparison.

```python
tic = time.time()
sol, info = solver.solve(max_iter=50) # Jitting solve
assert info[0]
toc = time.time()
jit_time = toc - tic
```

Now, we will demonstrate the comparison between the fast and slow implementations. The major reason that we achieve much faster simulation speed is through the Jitted functions that run extremely fast on the GPU. The problem is interlacing CPU native functions with GPU ones. In this loop, we incrementally solve 25 problems between tractions of 0. and 0.1 by reinitializing the dirichlet boundary conditions at each step through the `set_dirichlet_bc_info` method. This means that we are recomputing the boundary data at each step.

```python
vals = np.linspace(0., 0.1, 25)
toc = time.time()
sols = []
for v in vals:
    dirichlet_bcs = set_bcs(v)
    problem.set_dirichlet_bc_info(dirichlet_bcs)
    solver.initial_guess = sol
    sol, info = solver.solve(max_iter=50)
    sols.append(sol)
tic = time.time()
slow_set = tic - toc
```

The more clever approach is to recognize that the `bc_inds` remain fixed for a given protocol. Thus, we only need to appropriately scale the data if we set the initial values to magnitude of 1.

```python
dirichlet_bc_info = set_bcs(1.)
problem.set_dirichlet_bc_info(dirichlet_bc_info)
bc_inds, bc_vals = problem.get_boundary_data()
```

Now we use the `set_bc_vals` to just replace the current bc_vals by the specified ones. To allow for more general cases, the entire bc_vals vector must be the input of the function. Now, nearly all the computation remains on the GPU.

```python
vals = np.linspace(0., 0.1, 25)
toc = time.time()
sols = []
for v in vals:
    problem.set_bc_vals(v * bc_vals)
    solver.initial_guess = sol
    sol, info = solver.solve(max_iter=50)
    sols.append(sol)
tic = time.time()
fast_set = tic - toc
```

We then print the final values to check the times.

```python
print(f"Jit time: {jit_time:.2f} seconds")
print(f"Slow set time: {slow_set:.2f} seconds")
print(f"Fast set time: {fast_set:.2f} seconds")
```

```
Jit time: 3.34 seconds
Slow set time: 1.97 seconds
Fast set time: 1.49 seconds
```

Now we see time comparisons for a jitting solve, jitted slow solves, and jitted fast solves. The jit time demonstrates the power of solving sequential problems, so when jitted, we can solve 25 problems in half the time as one solve where jitting occurs. Also, we achieved a 25\% speedup just by noting that we should update the values instead of reinitializing. While a cherry picked example, these types of issues can easily sneak into implementations if not scrutinizing the code appropriately. This is especially important in the context of inverse problems where you may want to run 100s of loops for the optimization, and the time saved becomes much more important.
