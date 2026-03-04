# Import some useful modules.
import jax
import jax.numpy as np
import os
from pathlib import Path

from cardiax import box_mesh
from cardiax import Problem
from cardiax import Newton_Solver
from cardiax import FiniteElement

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
    
class LiveTraction(HyperElasticity):

    def get_surface_maps(self):

        def surface_map(u, u_grad, x, t, n):
            F = u_grad + np.eye(3)
            F_invT = np.linalg.inv(F).T
            J = np.linalg.det(F)
            return t[0] * J * F_invT @ n 

        return {"u": {"right": surface_map}}
    
class Deadload(HyperElasticity):

    def get_surface_maps(self):

        def surface_map(u, u_grad, x, t, n):
            return t[0] * n

        return {"u": {"right": surface_map}}

# Specify mesh-related information (first-order hexahedron element).
mesh = box_mesh(Nx=10, Ny=10, Nz=10, Lx=1., Ly=1., Lz=1.)
fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = "hexahedron", gauss_order = 1)

# Define boundary locations.
def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1., atol=1e-5)

# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.

bc1 = [[bottom] * 3, [0, 1, 2],
        [zero_dirichlet_val] * 3]
dirichlet_bc_info = {"u": [bc1]}
location_fns = {"u": {"right": right}}

problem_live = LiveTraction({"u": fe},
                          dirichlet_bc_info=dirichlet_bc_info,
                          location_fns=location_fns)
problem_dead = Deadload({"u": fe},
                          dirichlet_bc_info=dirichlet_bc_info,
                          location_fns=location_fns)

solver_live = Newton_Solver(problem_live, np.zeros((problem_live.num_total_dofs_all_vars)))
solver_dead = Newton_Solver(problem_dead, np.zeros((problem_dead.num_total_dofs_all_vars)))

forces = np.linspace(0, 1., 21, endpoint=True)
normals = fe.get_surface_normals(right)

problem_live.set_internal_vars_surfaces({"u": {"right": {"t": np.array([forces[-1]]), "n": normals}}})
sol1 = solver_live.solve(max_iter=50)[0]

problem_dead.set_internal_vars_surfaces({"u": {"right": {"t": np.array([forces[-1]]), "n": normals}}})
sol2 = solver_dead.solve(max_iter=50)[0]


# sols = []
# for f in forces:
#     problem.set_internal_vars_surfaces({"u": {"top": {"t": np.array([f])}}})
#     sol, info = solver.solve(max_iter=50)
#     assert info[0]
#     solver.initial_guess = sol
#     sols.append(sol)

if plotting := True:
    fig_dir = Path("../../../docs/figures/Introductory/neohookean/")
    import pyvista as pv
    import numpy as onp

    pl = pv.Plotter(shape=(1, 2), off_screen=True)
    mesh.point_data["sol1"] = onp.array(sol1).reshape(-1, fe.dim)
    mesh.point_data["sol2"] = onp.array(sol2).reshape(-1, fe.dim)
    warped1 = mesh.warp_by_vector("sol1", factor=1.)
    warped2 = mesh.warp_by_vector("sol2", factor=1.)
    pl.subplot(0, 0)
    pl.add_mesh(warped1, scalars="sol1")
    pl.subplot(0, 1)
    pl.add_mesh(warped2, scalars="sol2")
    pl.screenshot(fig_dir / "cube_live.png")
    pl.close()
