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
            c, a, b = 3., 2., 1.
            K = 300.
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
            return np.array([0., 0., -t[0]])
                
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

fibers1 = np.full((mesh.points.shape[0], 3), np.array([0., 0., 1.]))
fibers1 = fe.convert_dof_to_quad(fibers1)
fibers2 = np.full((mesh.points.shape[0], 3), np.array([0., np.sin(np.pi/6), np.cos(np.pi/6)]))
fibers2 = fe.convert_dof_to_quad(fibers2)

solver = Newton_Solver(problem, np.zeros((problem.num_total_dofs_all_vars)))

forces = np.linspace(0, 2., 41, endpoint=True)
sols1 = []
sols2 = []

problem.set_internal_vars({"u": {"f": fibers1}})
for i, f in enumerate(forces):
    problem.set_internal_vars_surfaces({"u": {"top": {"t": np.array([f])}}})
    sol, info = solver.solve(max_iter=50)
    assert info[0]
    solver.initial_guess = sol
    sols1.append(sol)

solver.initial_guess = np.zeros((problem.num_total_dofs_all_vars))
problem.set_internal_vars({"u": {"f": fibers2}})
for i, f in enumerate(forces):
    problem.set_internal_vars_surfaces({"u": {"top": {"t": np.array([f])}}})
    sol, info = solver.solve(max_iter=50)
    assert info[0]
    solver.initial_guess = sol
    sols2.append(sol)

if plotting := True:
    fig_dir = Path("../../figures/Intermediate/anisotropic_materials/")
    os.makedirs(fig_dir, exist_ok=True)
    import pyvista as pv
    import numpy as onp

    pl = pv.Plotter(shape=(1, 2), off_screen=True)
    mesh.point_data["sol1"] = onp.array(sols1[0]).reshape(-1, fe.dim)
    mesh.point_data["sol2"] = onp.array(sols2[0]).reshape(-1, fe.dim)
    mesh.cell_data["fiber1"] = onp.array(fibers1[:, 0, :]).reshape(-1, 3)
    mesh.cell_data["fiber2"] = onp.array(fibers2[:, 0, :]).reshape(-1, 3)
    warped1 = mesh.warp_by_vector("sol1", factor=1.)
    warped2 = mesh.warp_by_vector("sol2", factor=1.)
    glyph1 = warped1.glyph(orient="fiber1", scale=False, 
                               tolerance=0.05, factor=0.1)
    glyph2 = warped2.glyph(orient="fiber2", scale=False, 
                               tolerance=0.05, factor=0.1)

    pl.subplot(0, 0)
    pl.add_mesh(warped1, reset_camera=False, opacity=0.5,
                color="white", show_edges=True)
    pl.add_mesh(glyph1, color="red", reset_camera=False)
    pl.subplot(0, 1)
    pl.add_mesh(warped2, reset_camera=False, opacity=0.5,
                color="white", show_edges=True)
    pl.add_mesh(glyph2, color="red", reset_camera=False)
    pl.open_gif(fig_dir / "cyl_movie.gif")

    for i in range(len(sols1)):
        pl.clear()
        pl.subplot(0, 0)
        mesh.point_data["sol1"] = onp.array(sols1[i]).reshape(-1, fe.dim)
        warped1 = mesh.warp_by_vector("sol1", factor=1.)
        glyph1 = warped1.glyph(orient="fiber1", scale=False, 
                               tolerance=0.05, factor=0.1)
        pl.add_mesh(warped1, reset_camera=False, opacity=0.5,
                    color="white", show_edges=True)
        pl.add_mesh(glyph1, color="red", reset_camera=False)
        pl.add_title(f"force = {forces[i]:.3f}")
        pl.subplot(0, 1)
        mesh.point_data["sol2"] = onp.array(sols2[i]).reshape(-1, fe.dim)
        warped2 = mesh.warp_by_vector("sol2", factor=1.)
        glyph2 = warped2.glyph(orient="fiber2", scale=False, 
                               tolerance=0.05, factor=0.1)
        pl.add_mesh(warped2, reset_camera=False, opacity=0.5,
                    color="white", show_edges=True)
        pl.add_mesh(glyph2, color="red", reset_camera=False)
        pl.add_title(f"force = {forces[i]:.3f}")
        pl.write_frame()
    pl.close()
