# Import some useful modules.
import jax
import jax.numpy as np
import time
import functools as fctls

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
    
    def get_surface_maps(self):

        def surface_map(u, u_grad, x, t):
            return np.array([t[0], 0., 0.])
                
        return {"u": {"top": surface_map}}

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

def val(value, point):
    return value

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

problem = HyperElasticity({"u": fe},
                          dirichlet_bc_info=dirichlet_bc_info)
solver = Newton_Solver(problem, np.zeros((problem.num_total_dofs_all_vars)))

tic = time.time()
sol, info = solver.solve(max_iter=50) # Jitting solve
assert info[0]
toc = time.time()
jit_time = toc - tic

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

dirichlet_bc_info = set_bcs(1.)
problem.set_dirichlet_bc_info(dirichlet_bc_info)
bc_inds, bc_vals = problem.get_boundary_data()

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

print(f"Jit time: {jit_time:.2f} seconds")
print(f"Slow set time: {slow_set:.2f} seconds")
print(f"Fast set time: {fast_set:.2f} seconds")

if plotting := True:
    from pathlib import Path
    fig_dir = Path("../../../docs/figures/Introductory/neohookean/")
    import pyvista as pv
    import numpy as onp

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = onp.array(sols[0]).reshape(-1, fe.dim)
    warped = mesh.warp_by_vector("sol", factor=1.)

    pl.add_mesh(warped)
    pl.camera_position = 'xy'
    pl.camera.roll += 90
    pl.camera.azimuth += 30
    pl.camera.elevation += 30.
    pl.open_gif(fig_dir / "dirch_slow_movie.gif")

    for i, s in enumerate(sols):
        pl.clear()
        mesh.point_data["sol"] = onp.array(s).reshape(-1, fe.dim)
        warped = mesh.warp_by_vector("sol", factor=1.)
        pl.add_title(f"disp = {vals[i]:.3f}")
        pl.add_mesh(warped, scalars="sol", reset_camera=False)
        pl.write_frame()
    pl.close()
