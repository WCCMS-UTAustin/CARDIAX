# Import some generally useful packages.
import jax.numpy as np
import jax
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

        def tensor_map(u_grad, D):
            return u_grad @ D

        return tensor_map
    
    # Define the source term f
    # For the Poisson problem, using gaussian here
    def get_mass_map(self):
        def mass_map(u, u_grad, x, D):
            val = -3 * np.exp(-5 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / (2 * 0.1 ** 2))
            return np.array([val])
        return mass_map

    # Define potential surface kernels
    # Just sinusoidal here
    def get_surface_maps(self):
        def surface_map1(u, u_grad, x, a):
            return -np.sin(2 * a*x[0]*np.pi).reshape(1,)

        def surface_map2(u, u_grad, x, a):
            return np.sin(2 * a*x[0]*np.pi).reshape(1,)

        return {"u": {"bottom": surface_map1, "top": surface_map2}}
    
# Create the mesh and FE field
Lx, Ly = 1., 1.
mesh = rectangle_mesh(Nx=50, Ny=50, Lx=Lx, Ly=Ly, ele_type="quad8")
fe = FiniteElement(mesh, vec=1, dim=2, ele_type="quad8", gauss_order=2)

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

### Creates variables to solve over
num_frames = 51
a_values = np.linspace(-1, 1, num_frames, endpoint=True)
a_values = np.hstack([a_values, a_values[::-1]])

@jax.vmap
def D_vals(t):
    return np.array([[1., .5*t], [.5*t, 1.]]).reshape(1, 2, 2)

D_values = D_vals(a_values)

# Solve the problem
toc_jit = time.time()
problem.set_internal_vars({"u": {"D": D_values[0]}})
problem.set_internal_vars_surfaces({"u": {"bottom": {"a": np.array([a_values[0]])}, "top": {"a": np.array([a_values[0]])}}})
sol0, info = solver.solve(atol=1e-6)
assert info[0]
tic_jit = time.time()

sols = []
toc = time.time()
for n in range(len(a_values)):
    problem.set_internal_vars({"u": {"D": D_values[n]}})
    problem.set_internal_vars_surfaces({"u": {"bottom": {"a": np.array([a_values[n]])}, "top": {"a": np.array([a_values[n]])}}})
    sol, info = solver.solve()
    assert info[0]
    sols.append(onp.array(sol))
tic = time.time()

print("Initial solve time:", tic_jit - toc_jit)
print("Total solve time for all frames:", tic - toc)
print("Average time per frame:", (tic - toc)/(len(a_values)))

if plotting := True:
    import pyvista as pv

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = sol0.reshape(-1,)
    warped = mesh.warp_by_scalar("sol", factor=1.)
    pl.add_mesh(warped, scalars="sol", cmap="inferno", show_edges=True)
    pl.screenshot("../../../docs/figures/Introductory/poisson/poisson_initial.png")
    pl.close()

    pl = pv.Plotter(off_screen=True)
    mesh.point_data["sol"] = sols[0].reshape(-1,)
    warped = mesh.warp_by_scalar("sol", factor=1.)

    pl.add_mesh(warped, scalars="sol", cmap="inferno", show_edges=True)
    pl.open_gif("../../../docs/figures/Introductory/poisson/poisson_movie.gif")

    for i, s in enumerate(sols):
        pl.clear()
        mesh.point_data["sol"] = s.reshape(-1,)
        warped = mesh.warp_by_scalar("sol", factor=1.)
        pl.add_title(f"a = {a_values[i]:.2f}")
        pl.add_mesh(warped, scalars="sol", reset_camera=False, cmap="inferno", show_edges=True)
        pl.write_frame()
    pl.close()

