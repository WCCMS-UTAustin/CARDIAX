# Import some generally useful packages.
import jax.numpy as np
import numpy as onp
import time

from cardiax import rectangle_mesh
from cardiax import FiniteElement, Problem, Newton_Solver

# Define the Poisson problem
class Poisson(Problem):

    # This defines the kernel
    # \int \nabla u \cdot \nabla v dx
    # the "\cdot \nabla v" is fixed, so only provide the \nabla u
    def get_tensor_map(self):
        return lambda u_grad: u_grad
    
    # Define potential surface kernels
    # Just sinusoidal here
    def get_surface_maps(self):
        def surface_map1(u, u_grad, x):
            return -np.sin(2 * x[0] * np.pi).reshape(1,)

        def surface_map2(u, u_grad, x):
            return np.sin(2 * x[0] * np.pi).reshape(1,)

        return {"u": {"bottom": surface_map1, "top": surface_map2}}
    
# Create the mesh and FE field
Lx, Ly = 1., 1.
mesh = rectangle_mesh(Nx=50, Ny=50, Lx=Lx, Ly=Ly)
fe = FiniteElement(mesh, vec=1, dim=2, ele_type="quad", gauss_order=1)

# Define boundary locations.
def left(point):
    return np.isclose(point[0], 0., atol=1e-5)

def right(point):
    return np.isclose(point[0], Lx, atol=1e-5)

def bottom(point):
    return np.isclose(point[1], 0., atol=1e-5)

def top(point):
    return np.isclose(point[1], Ly, atol=1e-5)

# Define boundary values to assign (homogeneous)
def zero_bc(point):
    return 0.

# Combine BC info
bc_left = [[left], [0], [zero_bc]]
bc_right = [[right], [0], [zero_bc]]
dirichlet_bc_info = {"u": [bc_left, bc_right]}
location_fns = {"u": {"bottom": bottom, "top": top}}

problem = Poisson({"u": fe}, dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)

# Create instance of Newton_Solver
solver = Newton_Solver(problem, np.zeros((len(mesh.points), 1)))

# Solve the problem
toc = time.time()
sol, info = solver.solve(atol=1e-6)
assert info[0]
print(f"Poisson solved in {time.time() - toc:.2f} seconds.")

if plotting := True:
    import pyvista as pv

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = sol.reshape(-1,)
    warped = mesh.warp_by_scalar("sol", factor=1.)
    pl.add_mesh(mesh, cmap="inferno", color="white", opacity=0.5)
    pl.add_mesh(warped, cmap="inferno", show_edges=True)
    pl.screenshot("../../../docs/figures/Introductory/poisson/poisson_surface.png")
    pl.close()
