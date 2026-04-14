
import pyvista as pv
pv.OFF_SCREEN = True

from cardiax import ProblemManager

input_file = "cube_demo.yaml"

problem_manager = ProblemManager.from_yaml(input_file)
sol, info = problem_manager.solve_problem()