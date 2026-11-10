"""
Tests of quadrature rules on tetrahedra and hexahedra.

Considerations when creating new tests:
* If the integration domain in changed, make sure `integrate_both_ways`
  is able to incorporate it
* Choose a high-order polynomial function that will be exactly integrated
* Choose a low-order polynomial that will *not* be exactly integrated
* Count up the flops in the polynomials to pass to the assertions
"""
import unittest
import numpy as np
import meshio, pyvista as pv
from cardiax.Lagrange.fe import FiniteElement
import sympy


x, y, z = sympy.symbols("x,y,z")


def integrate_both_ways(fe, func):
    # Numerical
    f = sympy.lambdify((x, y, z), func, "numpy")
    qp = np.asarray(fe.get_physical_quad_points())
    JxW = np.asarray(fe.JxW)
    numerical_soln = np.sum(f(qp[...,0], qp[...,1], qp[...,2]) * JxW)
    
    #
    # Symbolic
    #
    # Introspect element for integration bounds
    if fe.ele_type == "tetrahedron":
        bounds_spec = [(x, 0, 1 - y - z), (y, 0, 1 - z), (z, 0, 1)]
    elif fe.ele_type == "hexahedron":
        bounds_spec = [(x, 0, 1), (y, 0, 1), (z, 0, 1)]
    else:
        raise ValueError(
            f"test_quadrature symbolic integration encountered unknown ele_type = {fe.ele_type}"
        )

    # Perform symbolic integration, get float
    true_soln = sympy.integrate(sympy.integrate(sympy.integrate(
        func,
        bounds_spec[0]), bounds_spec[1]), bounds_spec[2]
    ).evalf()
    assert isinstance(true_soln, sympy.Float), f"SymPy issue in test of {func}: true solution is type {type(true_soln)}"
    true_soln = np.array(true_soln, dtype=numerical_soln.dtype) # Cast to NumPy

    return numerical_soln, true_soln


def assert_is_integrated_accurately(fe, func, integrand_flops):
    numerical_soln, true_soln = integrate_both_ways(fe, func)

    rtol = (2*fe.num_quads - 1 + fe.num_quads*integrand_flops)*np.finfo(numerical_soln.dtype).eps # Num flops * machine eps
    assert np.isclose(
        numerical_soln, true_soln,
        rtol=rtol,
        atol=0
    ), f"Numerical {numerical_soln} != Symbolic {true_soln} (within {rtol})"


def assert_is_not_integrated_accurately(fe, func, integrand_flops):
    numerical_soln, true_soln = integrate_both_ways(fe, func)

    rtol = (2*fe.num_quads - 1 + fe.num_quads*integrand_flops)*np.finfo(numerical_soln.dtype).eps # Num flops * machine eps
    assert not np.isclose(
        numerical_soln, true_soln,
        rtol=rtol,
        atol=0
    ), f"Numerical {numerical_soln} == Symbolic {true_soln} (within {rtol})"


class testTetrahedron(unittest.TestCase):

    def setUp(self) -> None:
        points = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        # Only need this for geometry; basis functions will be assembled by hand
        self.mesh = pv.from_meshio(meshio.Mesh(points, [("tetra", np.array([[0,1,2,3]]))]))

    def testQuadratureOrder1(self) -> None:
        fe = FiniteElement(self.mesh, vec=1, dim=3, ele_type="tetrahedron", quadrature_order=1)

        # Accurately integrates linear function
        func = x+y+z
        assert_is_integrated_accurately(fe, func, 2)

        # Does not accurately integrate quadratic function
        func = x*(1-x)+(0.5-y)*(1-y)+z*(0.5-z)
        assert_is_not_integrated_accurately(fe, func, 9)

    def testQuadratureOrder2(self) -> None:
        fe = FiniteElement(self.mesh, vec=1, dim=3, ele_type="tetrahedron", quadrature_order=2)

        # Accurately integrates quadratic function
        func = x*(1-x)+(0.5-y)*(1-y)+z*(0.5-z)
        assert_is_integrated_accurately(fe, func, 9)

        # Does not accurately integrate cubic function
        third = sympy.S("1/3")
        func = x*(third - x)*(2*third - x)
        assert_is_not_integrated_accurately(fe, func, 5)

    def testQuadratureOrder3(self) -> None:
        fe = FiniteElement(self.mesh, vec=1, dim=3, ele_type="tetrahedron", quadrature_order=3)

        # Accurately integrates cubic function
        third = sympy.S("1/3")
        func = x*(third - x)*(2*third - x)
        assert_is_integrated_accurately(fe, func, 5)

        # Does not accurately integrate quartic function
        func = x*(0.25 - x)*(0.5 - x)*(0.75 - x)
        assert_is_not_integrated_accurately(fe, func, 6)


class testHexahedron(unittest.TestCase):

    def setUp(self) -> None:
        points = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                   [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
        # Only need this for geometry; basis functions will be assembled by hand
        self.mesh = pv.from_meshio(meshio.Mesh(points,
            [("hexahedron", np.array([[0,1,2,3,4,5,6,7]]))]))

    def testQuadratureOrder1(self) -> None:
        fe = FiniteElement(self.mesh, vec=1, dim=3, ele_type="hexahedron", quadrature_order=1)

        # Accurately integrates trilinear function
        func = x*y*z
        assert_is_integrated_accurately(fe, func, 2)

        # Does not accurately integrate quadratic function
        func = x*(1-x)
        assert_is_not_integrated_accurately(fe, func, 2)

    def testQuadratureOrder2(self) -> None:
        fe = FiniteElement(self.mesh, vec=1, dim=3, ele_type="hexahedron", quadrature_order=2)

        # Accurately integrates triquadratic function
        func = x*(1-x)*(0.5-y)*(1-y)*z*(0.5-z)
        assert_is_integrated_accurately(fe, func, 9)

        # Does not accurately integrate quartic function
        # - Gauss-Jacobi quadrature picks up an extra degree every-other
        func = x*(0.25 - x)*(0.5 - x)*(0.75 - x)
        assert_is_not_integrated_accurately(fe, func, 6)

    def testQuadratureOrder3(self) -> None:
        fe = FiniteElement(self.mesh, vec=1, dim=3, ele_type="hexahedron", quadrature_order=3)

        # Accurately integrates tricubic function
        third = sympy.S("1/3")
        func = x*(third - x)*(2*third - x)*y*(third - y)*(2*third - y)*z*(third - z)*(2*third - z)
        assert_is_integrated_accurately(fe, func, 17)

        # Does not accurately integrate quintic function
        fifth = sympy.S("1/5")
        func = x*(fifth - x)*(2*fifth - x)*(3*fifth - x)*(4*fifth - x)
        assert_is_not_integrated_accurately(fe, func, 11)
