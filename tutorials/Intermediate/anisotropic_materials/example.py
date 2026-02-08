# Import some useful modules.
import jax
import jax.numpy as np
import os
from pathlib import Path

from cardiax import cylinder_mesh
from cardiax import FiniteElement, Problem, Newton_Solver
from cardiax import get_F

class HyperElasticity(Problem):

    def get_tensor_map(self):

        def psi(F, f):
            c, a, b = 3., 2., 0.5
            K = 30.
            J = np.linalg.det(F)
            I1 = np.trace(F.T @ F)
            I4 = f.T @ F @ f
            matrix = c * (I1 - 3.)
            fiber = a / (2 * b) * (np.exp(b * (I4**2 - 1.)**2) - 1.)
            comp = K / 2 * ( (J**2 - 1)/2 - np.log(J) )
            return matrix + fiber + comp

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, f):
            I = np.eye(u_grad.shape[0])
            F = u_grad + I
            P = P_fn(F, f)
            return P

        return first_PK_stress
    
    def get_surface_maps(self):

        def surface_map(u, u_grad, x, t):
            return np.array([0., 0., t[0]])
                
        return {"u": {"top": surface_map}}

# Specify mesh-related information (first-order hexahedron element).
mesh = cylinder_mesh(height=1., radius=0.25, cl_max=.05, cl_min=.01)
fe = FiniteElement(mesh, vec = 3, dim = 3, ele_type = "tetrahedron", gauss_order = 1)

# Define boundary locations.
def bottom(point):
    return np.isclose(point[2], 0., atol=1e-5)

def top(point):
    return np.isclose(point[2], 1., atol=1e-5)

# Define Dirichlet boundary values.
def zero_dirichlet_val(point):
    return 0.

bc1 = [[bottom] * 3, [0, 1, 2],
        [zero_dirichlet_val] * 3]
dirichlet_bc_info = {"u": [bc1]}
location_fns = {"u": {"top": top}}

problem = HyperElasticity({"u": fe},
                          dirichlet_bc_info=dirichlet_bc_info,
                          location_fns=location_fns)

fibers = np.full((mesh.points.shape[0], 3), np.array([0., 0., 1.]))
problem.set_internal_vars({"u": {"f": fibers}})

solver = Newton_Solver(problem, np.zeros((problem.num_total_dofs_all_vars)))

forces = np.linspace(0, 1., 21, endpoint=True)

sols = []
fibers_list = []
for i, f in enumerate(forces):
    problem.set_internal_vars_surfaces({"u": {"top": {"t": np.array([f])}}})
    sol, info = solver.solve(max_iter=50)
    assert info[0]
    solver.initial_guess = sol
    sols.append(sol)

    Fs = get_F(fe, sol).mean(axis=1)
    fibers_cell = fe.convert_dof_to_quad(fibers).mean(axis=1)
    fibers_new = np.einsum("ijk, ik -> ik", Fs, fibers_cell)    
    fibers_list.append(fibers_new)

for i in range(60):
    fiber = np.array([np.sin(i/30 * np.pi), 0., np.cos(i/30 * np.pi)])
    fibers_list.append(fiber)
    fibers_new = np.full((mesh.points.shape[0], 3), fiber)
    problem.set_internal_vars({"u": {"f": fibers_new}})
    sol, info = solver.solve(max_iter=50)
    assert info[0]
    solver.initial_guess = sol
    sols.append(sol)
    
    Fs = get_F(fe, sol).mean(axis=1)
    fibers_cell = fe.convert_dof_to_quad(fibers_new).mean(axis=1)
    fibers_new = np.einsum("ijk, ik -> ik", Fs, fibers_cell)
    fibers_list.append(fibers_new)

    mesh.point_data["sol"] = np.array(sol).reshape(-1, fe.dim)
    mesh.cell_data["fiber"] = np.array(fibers_new).reshape(-1, 3)
    mesh.save(f"cyl_{i}.vtk")

exit()
if plotting := True:
    fig_dir = Path("../../figures/Intermediate/anisotropic_materials/")
    os.makedirs(fig_dir, exist_ok=True)
    import pyvista as pv
    import numpy as onp

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = onp.array(sols[0]).reshape(-1, fe.dim)
    mesh.point_data["fiber"] = onp.array(fibers_list[0]).reshape(-1, 3)
    warped = mesh.warp_by_vector("sol", factor=1.)

    pl.add_mesh(warped)
    pl.open_gif(fig_dir / "cyl_movie.gif")

    for i, s in enumerate(sols):
        pl.clear()
        mesh.point_data["sol"] = onp.array(s).reshape(-1, fe.dim)
        mesh.point_data["fiber"] = onp.array(fibers_list[i]).reshape(-1, 3)
        warped = mesh.warp_by_vector("sol", factor=1.)
        glyph = warped.glyph(orient="fiber", scale=False, factor=0.1)
        pl.add_title(f"force = {forces[i]:.3f}")
        pl.add_mesh(warped, reset_camera=False)
        pl.add_mesh(glyph, color="red", reset_camera=False)
        pl.write_frame()
    pl.close()
