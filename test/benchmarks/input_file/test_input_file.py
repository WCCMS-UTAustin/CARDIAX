import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import meshio
import os
import unittest

from cardiax import ProblemManager

jax.config.update("jax_enable_x64", True)

class Test(unittest.TestCase):
    def test_solve_problem(self):
        """ Test hyperelasticity analytic solution for shear
        """

        # input_file = "test/benchmarks/input_file/inputs.yaml"
        input_file = "inputs.yaml"
        manager = ProblemManager.from_yaml(input_file)
        sol, info = manager.solve_problem()
        assert info[0]

if __name__ == '__main__':
    unittest.main()