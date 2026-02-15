# Import some generally useful packages.
import jax.numpy as np
import time

from cardiax import rectangle_mesh
from cardiax import FiniteElement, Problem, Newton_Solver

# Define the Poisson problem
class Poisson(Problem):

    # Defines the contraction between
    # \int \nabla u \cdot \nabla v dx
    # You're entering the "\nabla u" part, the "\cdot \nabla v" is fixed
    def get_tensor_map(self):
        return lambda u_grad: u_grad
    
    # Define the source term f
    # For the Poisson problem, using eigenfunction here
    def get_mass_map(self):
        def mass_map(u, u_grad, x):
            val = -10 * np.array([np.sin(2 * x[0] * np.pi) * np.sin(2 * x[1] * np.pi)])
            return val
        return mass_map
    
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
bc_top = [[top], [0], [zero_bc]]
bc_bottom = [[bottom], [0], [zero_bc]]
dirichlet_bc_info = {"u": [bc_left, bc_right, bc_top, bc_bottom]}

problem = Poisson({"u": fe}, dirichlet_bc_info=dirichlet_bc_info)

# Create instance of Newton_Solver
solver = Newton_Solver(problem, np.zeros((len(mesh.points), 1)))

# Solve the problem
toc = time.time()
sol, info = solver.solve(atol=1e-6)
assert info[0]
tic = time.time()
print(f"Poisson problem solved in {tic - toc:.2f} seconds.")

if plotting := True:
    import jax
    import pyvista as pv

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(mesh, show_edges=True, color="white", opacity=0.5)
    pl.screenshot("../../figures/Introductory/poisson/poisson_mesh.png")
    pl.close()

    def f(x):
        return -10 * np.array([np.sin(2 * x[0] * np.pi) * np.sin(2 * x[1] * np.pi)])
    fs = jax.vmap(f)(mesh.points)

    pl = pv.Plotter(shape=(1, 2), off_screen=True)
    pl.subplot(0, 0)
    mesh.point_data["fs"] = fs.reshape(-1,)
    warped = mesh.warp_by_scalar("fs", factor=0.025)
    pl.add_mesh(warped, show_edges=True, cmap="viridis",
                scalars="fs", copy_mesh=True,
                scalar_bar_args={"title": "f (scaled down)"},
                clim=[1.5 * fs.min(), 1.5 * fs.max()])
    pl.add_title("Source term f \n (scaled down)")
    pl.subplot(0, 1)
    mesh.point_data["sol"] = sol.reshape(-1,)
    warped = mesh.warp_by_scalar("sol", factor=1.)
    pl.add_mesh(warped, show_edges=True, cmap="inferno", 
                scalars="sol", copy_mesh=True,
                scalar_bar_args={"title": "u"},
                clim=[1.5 * sol.min(), 1.5 * sol.max()])
    pl.add_title("Solution u")
    pl.screenshot("../../figures/Introductory/poisson/poisson_isotropic.png")
    pl.close()