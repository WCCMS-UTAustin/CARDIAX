
import pyvista as pv
pv.OFF_SCREEN = True

import jax.numpy as np

from cardiax import ProblemManager

# The Newton solver normally starts from zero and lifts the Dirichlet values in
# through the linear solve, so collapsed elements never reach the material model.
# To show the safeguard, start from an initial guess that only has the boundary
# values assigned: the top and bottom element layers (height 0.1) are compressed
# by 0.1 and start at J = 0. Without the clamp this gives NaNs immediately.
for input_file in ["unclamped_NH.yaml", "clamped_NH.yaml"]:

    problem_manager = ProblemManager.from_yaml(input_file)
    solver = problem_manager.solver
    solver.initial_guess = solver.assign_bc(np.zeros(problem_manager.problem.num_total_dofs_all_vars))

    sol, info = problem_manager.solve_problem()
    converged, num_iters, res_vec = info[0], info[1], info[-1]
    print(f"{input_file}: converged = {converged}, Newton iterations = {num_iters}, "
          f"NaNs in residual = {bool(np.isnan(res_vec).any())}")
