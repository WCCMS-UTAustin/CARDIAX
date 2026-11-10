# Kenneth Meyer
# 10/19/26
# Tests handling of internal variables
#
# run me via 'pytest test_internal_vars.py'

import numpy as onp
import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)

import pytest
import numpy.testing as onptest

from cardiax import box_mesh, FiniteElement, Problem

# sample problem class to test internal variable ordering on
class InteralVarTester(Problem):
    def get_tensor_map(self):
        """
        tensor map with fiber, sheet, and normals, ordered (s, f, n) so the
        signature is non-alphabetical. The matrix product s @ f @ n does not
        commute, so any mix-up in argument order changes the output.
        """
        def tensor_map(u_grad, s, f, n):
            return s @ f @ n

        return tensor_map

# create a problem class with specific kernel type required
# for the test (tensor map with arguments)
def _get_problem():
    mesh = box_mesh(Nx=2, Ny=3, Nz=4)
    fe = FiniteElement(mesh, vec=3, dim=3, ele_type="hexahedron", quadrature_order=2)
    problem = InteralVarTester({'u': fe})
    return problem

# three distinct 3x3 matrices, passed as constant internal variables (shape (1, 3, 3))
def _get_matrices():
    rng = onp.random.default_rng(0)
    A, B, C = (np.array(rng.standard_normal((3, 3))) for _ in range(3))

    # sanity check: every argument permutation gives a different product,
    # otherwise a wrong ordering could pass by accident
    products = [X @ Y @ Z for X, Y, Z in [(A, B, C), (A, C, B), (B, A, C),
                                          (B, C, A), (C, A, B), (C, B, A)]]
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            assert not np.allclose(products[i], products[j])
    return A, B, C

def _call_tensor_map(problem, internal_vars):
    """ call the tensor map the same way split_and_compute_cell does: internal vars
    are looked up by name through problem.ordered_dict and passed positionally.
    Evaluated at the first cell / quad point, since the vars are constant.
    """
    tensor_map = problem.get_tensor_map()
    args = [internal_vars['u'][key][0, 0] for key in problem.ordered_dict['u']]
    return tensor_map(np.zeros((3, 3)), *args)

def _check_tensor_map(problem, P_expected):
    eager = _call_tensor_map(problem, problem.internal_vars)
    # jit flattens the internal_vars dict and rebuilds it with sorted keys, so this
    # catches any reliance on dict ordering
    jitted = jax.jit(lambda iv: _call_tensor_map(problem, iv))(problem.internal_vars)
    onptest.assert_allclose(eager, P_expected, rtol=1e-12, atol=1e-12)
    onptest.assert_allclose(jitted, P_expected, rtol=1e-12, atol=1e-12)

# directly test each desired feature. re-generating the problem
# might not be desired but ok for now.
#
# subtests *could* be used here in the future.
@pytest.mark.parametrize("insertion_order", [("s", "f", "n"),   # same as signature
                                             ("f", "n", "s"),   # alphabetical
                                             ("n", "f", "s")])  # reversed
def test_exact_match(insertion_order):

    """ check if internal vars are properly set
    if there is an exact match
    """

    S, F, N = _get_matrices()
    values = {"s": S[None], "f": F[None], "n": N[None]}

    problem = _get_problem()
    problem.set_internal_vars({'u': {key: values[key] for key in insertion_order}})

    # names match the signature, so the signature order is used regardless of insertion order
    assert problem.ordered_dict['u'] == ["s", "f", "n"]
    _check_tensor_map(problem, S @ F @ N)

def test_ordering_fallback():

    """ check if internal vars are properly set
    if there is a mismatch in function signature
    and internal variable name

    make sure to design this test to check for
    behavior when JAX may re-order a dictionary - might need
    to jit something; see the following issue for more:

    https://github.com/jax-ml/jax/issues/24398
    """

    A, B, C = _get_matrices()

    # none of these names are in the signature (s, f, n), and the insertion order
    # is neither alphabetical (fiber, normal, sheet) nor the signature order, so
    # only a positional mapping by insertion order gives A @ B @ C
    problem = _get_problem()
    problem.set_internal_vars({'u': {"normal": A[None], "fiber": B[None], "sheet": C[None]}})

    assert problem.ordered_dict['u'] == ["normal", "fiber", "sheet"]
    _check_tensor_map(problem, A @ B @ C)

def test_wrong_number_of_vars():

    """ check that giving a different number of internal vars than the
    tensor map takes raises an error
    """

    S, F, N = _get_matrices()

    problem = _get_problem()
    with pytest.raises(AssertionError, match="take 3 internal variables"):
        problem.set_internal_vars({'u': {"s": S[None], "f": F[None]}})
