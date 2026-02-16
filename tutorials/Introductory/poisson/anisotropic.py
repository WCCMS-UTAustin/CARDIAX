# Import some generally useful packages.
import jax.numpy as np
import numpy as onp
import time

from cardiax import rectangle_mesh
from cardiax import FiniteElement, Problem, Newton_Solver

# Define the Poisson problem
class Poisson(Problem):

    # Defines the contraction between
    # \int \nabla u \cdot \nabla v dx
    # You're entering the "\nabla u" part, the "\cdot \nabla v" is fixed
    def get_tensor_map(self):

        def tensor_map(u_grad, D):
            return u_grad @ D

        return tensor_map
                
    # Define the source term f
    # For the Poisson problem, using eigenfunction here
    def get_mass_map(self):
        def mass_map(u, u_grad, x, D):
            val = -10 * np.exp(-5 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / (2 * 0.1 ** 2))
            return np.array([val])
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
I = np.array([[1., 0.], [0., 1.]])
A = np.array([[1., 1.], [0., 1.]])

# Create instance of Newton_Solver
solver = Newton_Solver(problem, np.zeros((len(mesh.points), 1)))

# Solve the problem
problem.set_internal_vars({"u": {"D": I}})
toc_I = time.time()
sol_I, info = solver.solve(atol=1e-6)
assert info[0]
tic_I = time.time()

problem.set_internal_vars({"u": {"D": A}})
toc_A = time.time()
sol_A, info = solver.solve(atol=1e-6)
assert info[0]
tic_A = time.time()

print("Isotropic solve time (including JIT): ", tic_I - toc_I)
print("Anisotropic solve time (JITTED): ", tic_A - toc_A)

if plotting := True:
    import pyvista as pv
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

    pl = pv.Plotter(shape=(1, 2), off_screen=True)
    pl.subplot(0, 0)
    mesh.point_data["sol"] = sol_I.reshape(-1,)
    pl.add_mesh(mesh, cmap="inferno", copy_mesh=True)
    pl.view_xy()
    pl.add_title("Isotropic", font_size=12)
    pl.subplot(0, 1)
    mesh.point_data["sol"] = sol_A.reshape(-1,)
    pl.add_mesh(mesh, cmap="inferno", copy_mesh=True)
    pl.view_xy()
    pl.add_title("Anisotropic", font_size=12)
    pl.screenshot("../../figures/Introductory/poisson/poisson_anisotropic.png")
    pl.close()
