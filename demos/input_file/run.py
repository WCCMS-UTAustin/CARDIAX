
import pyvista as pv
pv.OFF_SCREEN = True

from cardiax import FE_manager

input_file = "cube_demo.yaml"

FE_manager = FE_manager(input_file)
sol, info = FE_manager.solve_problem()

