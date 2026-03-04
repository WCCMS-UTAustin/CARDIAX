# Import some useful modules.
import jax
import jax.numpy as np

from cardiax import box_mesh
from cardiax import Problem
from cardiax import Newton_Solver
from cardiax import FiniteElement

class HyperElasticity(Problem):

    def get_tensor_map(self):

        def psi(F):
            E = 10.
            nu = 0.45
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
    
# Specify mesh-related information (first-order hexahedron element).
mesh = box_mesh(Nx=10, Ny=10, Nz=10, Lx=1., Ly=1., Lz=1.)
fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = "hexahedron", gauss_order = 1)

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], 1., atol=1e-5)

# Define Dirichlet boundary values.
def zero(point):
    return 0.

theta = np.pi / 4.
def val0(point):
    return (point[1]-.5)*(np.cos(theta)-1) - (point[2]-.5)*np.sin(theta)

def val1(point):
    return (point[1]-.5)*np.sin(theta) + (point[2]-.5)*(np.cos(theta)-1)

bc_left = [
    [left] * 3,
    [0, 1, 2],
    [zero, zero, zero]
    ]
bc_right = [
    [right] * 3, 
    [0, 1, 2],
    [zero, val0, val1]
    ]

dirichlet_bc_info = {"u": [bc_left, bc_right]}

problem = HyperElasticity({"u": fe},
                          dirichlet_bc_info=dirichlet_bc_info)

solver = Newton_Solver(problem, np.zeros((problem.num_total_dofs_all_vars)))
sol, info = solver.solve(max_iter=10)
assert info[0]



if plotting := True:
    from pathlib import Path
    fig_dir = Path("../../../docs/figures/Introductory/neohookean/")
    import pyvista as pv
    import numpy as onp

    pl = pv.Plotter(shape=(1, 2), off_screen=True)
    zero_sol = onp.zeros((problem.num_total_dofs_all_vars))
    bc_inds, bc_vals = problem.get_boundary_data()
    zero_sol[bc_inds] = bc_vals
    mesh.point_data["sol"] = onp.array(zero_sol).reshape(-1, fe.dim)
    warped = mesh.warp_by_vector("sol", factor=1.)
    pl.subplot(0, 0)
    pl.add_mesh(mesh, scalars=None, color="white", show_edges=True)
    pl.subplot(0, 1)
    pl.add_mesh(warped, scalars="sol", show_edges=True)
    pl.screenshot(fig_dir / "dirch_distorted.png")
    pl.close()

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = onp.array(sol).reshape(-1, fe.dim)
    warped = mesh.warp_by_vector("sol", factor=1.)
    pl.add_mesh(warped, scalars="sol")
    pl.screenshot(fig_dir / "dirch_cube.png")
    pl.close()