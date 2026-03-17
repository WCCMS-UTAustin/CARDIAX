import jax
import unittest
import os
from pathlib import Path

from cardiax import ProblemManager

jax.config.update("jax_enable_x64", True)

class Test(unittest.TestCase):
    def test_solve_problem(self):
        """ Test hyperelasticity solution with input file
        """
        problem_name = "input_file_test"

        parent_dir = Path(__file__).parent
        input_file = parent_dir / 'inputs.yaml'
        manager = ProblemManager.from_yaml(input_file)
        sol, info = manager.solve_problem()
        assert info[0]

if __name__ == '__main__':
    unittest.main()