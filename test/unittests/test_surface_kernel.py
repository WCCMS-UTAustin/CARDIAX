"""

Tests surface kernel, specifically with varying inputs,
in CARDIAX.

"""

import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import unittest

from cardiax import rectangle_mesh
from cardiax import FiniteElement, Problem

class SurfaceKernelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # create FE and problem object.

        return super().setUpClass()
    
    @unittest.skip("Not implemented yet")
    def test_single_traction(self):
        """ tests defining traction as a constant field.
        """

        # do something here

    @unittest.skip("Not implemented yet")
    def test_traction_cells(self):
        """ tests defining traction that is constant on each cell.
        """

        # do something here

    
    @unittest.skip("Not implemented yet")
    def test_traction_cells_nodes(self):
        """ tests defining traction at nodes on each cell facet.
        """

        # do something here

    # any other useful tests?


if __name__ == "__main__":
    unittest.main()