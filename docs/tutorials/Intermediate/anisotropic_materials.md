
# General Hyperelasticity

## Anisotropic Hyperelastic Materials

After going through the simpler Neo-Hookean example, we can look at fully anisotropic models based on fiber directions. For anisotropic material models, we add in a fiber field to direct the anisotropic modes of the strain energy. We can then have a transversly isotropic material defined by

$$
\Psi(\mathbf{F}) = c (I_1 - 3) + \frac{a}{2b} \left( e^{b(I_4 - 1)^2} - 1\right) + \frac{K}{2} \left( \frac{J^2 - 1}{2} - \ln(J) \right)
$$

with a bulk stiffness $c$, two parameters for the anisotropic component $a$ and $b$, and a incompressibility parameter $K$.

## Implementation

### Imports

First, we start with basic imports.

```python
import jax
import jax.numpy as np
import os

from cardiax import box_mesh
from cardiax import FiniteElement, Problem, Newton_Solver
```

### Finite Element Discretization

For the finite element discretization, we will create a beam with our box mesh.

```python
# Specify mesh-related information (first-order hexahedron element).
mesh = box_mesh(Nx=10, Ny=10, Nz=50, Lx=1., Ly=1., Lz=5.)
fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = "hexahedron", gauss_order = 1)
```

### Boundary conditions

The location functions are defined at the two faces of the beam. The bottom, we enforce zero movement in all three directions. The top face is where we apply the traction on the beam.

```python
# Define boundary locations.
def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)

def top(point):
    return np.isclose(point[2], 5., atol=1e-5)

# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.

bc1 = [[bottom] * 3, [0, 1, 2],
        [zero_dirichlet_val] * 3]
dirichlet_bc_info = {"u": [bc1]}
location_fns = {"u": {"top": top}}
```

### Problem Construction

To define the problem, we create the strain energy function $\Psi$ then takes derivatives of the strain energy with respect to the $\mathbf{F}$ to obtain $\mathbf{P}$. We also define the surface map to apply the traction in the x direction to move the beam towards the side.

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
    
    def get_surface_maps(self):

        def surface_map(u, u_grad, x, t):
            return np.array([t[0], 0., 0.])
                
        return {"u": {"top": surface_map}}

problem = HyperElasticity({"u": fe},
                          dirichlet_bc_info=dirichlet_bc_info,
                          location_fns=location_fns)

```

### Solver

The solver is then initiated with the problem and boundary conditions defined above. Line search is implemented and quite customizable if needed, so we can solve the final traction immediately. We can also perform incremental solves to generate the intermediate solutions for the gif shown below.

```python
solver = Newton_Solver(problem, np.zeros((problem.num_total_dofs_all_vars)))

forces = np.linspace(0, .025, 21, endpoint=True)

problem.set_internal_vars_surfaces({"u": {"top": {"t": np.array([forces[-1]])}}})
sol0 = solver.solve(max_iter=50)[0]

sols = []
for f in forces:
    problem.set_internal_vars_surfaces({"u": {"top": {"t": np.array([f])}}})
    sol, info = solver.solve(max_iter=50)
    assert info[0]
    solver.initial_guess = sol
    sols.append(sol)
```

![alt text](../../../tutorials/figures/Introductory/neohookean/beam_movie.gif)