"""

    Tests the configurable adjoint linear solver.

    Covers:
      * ``adjoint_solve`` solves the transpose of the forward system, which is
        what the reduced-gradient computation needs;
      * the default (``adjoint_solver is None``) reuses the forward solver, so
        behavior is unchanged for callers that never configure one;
      * ``set_adjoint_solver`` installs and clears a dedicated adjoint solver,
        and the adjoint solve actually honors it rather than silently falling
        back to the forward solver;
      * choosing an adjoint solver after a forward solve has already been
        compiled still takes effect.

"""

import numpy as onp
import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)

import unittest
from unittest.mock import MagicMock
import numpy.testing as onptest

from cardiax._solver import Solver_Base
from cardiax import Problem
import lineax as lx


def _get_VIJ_bcoo(A_np: onp.ndarray):
    """Re-organize a dense matrix into (data, row, col) arrays (row-major)."""
    V = onp.array([], dtype=onp.float64)
    I = onp.array([], dtype=onp.int32)
    J = onp.array([], dtype=onp.int32)
    n_row, n_col = A_np.shape
    ones_array = onp.ones(n_row, dtype=onp.int32)
    cols_array = onp.arange(n_col, dtype=onp.int32)
    for i in range(len(A_np)):
        V = onp.hstack((V, A_np[i]), dtype=onp.float64)
        I = onp.hstack((I, i * ones_array), dtype=onp.int32)
        J = onp.hstack((J, cols_array), dtype=onp.int32)
    return V, I, J


def _mock_problem(init_guess, I, J):
    mock_problem = MagicMock(Problem)
    mock_problem.num_total_dofs_all_vars = len(init_guess)
    mock_problem.I = np.array(I)
    mock_problem.J = J
    mock_problem.bc_inds = onp.array([], dtype=onp.int32)  # no BCs for this test
    mock_problem.internal_vars = {'u': []}
    mock_problem.internal_vars_surfaces = {'u': []}
    return mock_problem


class Test_AdjointSolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        onp.random.seed(0)
        n = 20
        # Deliberately NON-symmetric so that solving with the transposed
        # pattern is distinguishable from solving with the original one.
        # Diagonally dominant, so every Krylov method used here converges.
        cls.n = n
        cls.A_np = onp.random.rand(n, n) + n * onp.eye(n)
        cls.V, cls.I, cls.J = _get_VIJ_bcoo(cls.A_np)
        cls.b = onp.arange(1.0, n + 1.0)
        cls.x0 = onp.zeros(n)

    def _fresh_solver(self):
        """A solver over the test matrix, with no Dirichlet dofs."""
        init_guess = onp.zeros(self.n)
        solver = Solver_Base(_mock_problem(init_guess, self.I, self.J),
                             init_guess, precond='jacobi',
                             line_search_flag=False)
        # Populate the operator data (constructed with a dummy template).
        solver.A.matrix.data = self.V
        return solver

    # ----------------------------------------------------------------- #
    def test_default_adjoint_solver_is_none(self):
        """By default the adjoint reuses the forward solver (back-compat)."""
        assert self._fresh_solver().adjoint_solver is None

    def test_set_adjoint_solver_roundtrip(self):
        """set_adjoint_solver installs and clears the dedicated solver."""
        solver = self._fresh_solver()
        cg = lx.CG(rtol=1e-8, atol=1e-10, max_steps=100)
        solver.set_adjoint_solver(cg)
        assert solver.adjoint_solver is cg
        solver.set_adjoint_solver(None)
        assert solver.adjoint_solver is None

    def test_adjoint_solve_uses_transposed_pattern(self):
        """Passing swapped indices solves with A transposed, not with A."""
        solver = self._fresh_solver()
        x, err = solver.adjoint_solve(self.V, self.J, self.I, self.b, self.x0)

        assert float(err) < 1e-8
        onptest.assert_allclose(onp.asarray(x),
                                onp.linalg.solve(self.A_np.T, self.b),
                                rtol=1e-8, atol=1e-10)
        # The matrix is non-symmetric, so the forward solution differs; if the
        # transpose were being ignored this assertion would fail.
        assert not onp.allclose(onp.asarray(x),
                                onp.linalg.solve(self.A_np, self.b))

    def test_default_adjoint_matches_forward_solver(self):
        """With no adjoint solver set, the adjoint solves as jax_solve would."""
        solver = self._fresh_solver()
        adjoint_x, _ = solver.adjoint_solve(self.V, self.J, self.I,
                                            self.b, self.x0)
        # jax_solve on the transposed pattern is the historical adjoint path.
        forward_x, _ = solver.jax_solve(self.V, self.J, self.I,
                                        self.b, self.x0)
        onptest.assert_allclose(onp.asarray(adjoint_x), onp.asarray(forward_x),
                                rtol=1e-10, atol=1e-12)

    def test_adjoint_solve_honors_its_own_solver(self):
        """A deliberately starved adjoint solver degrades only the adjoint.

        The forward solver is left at its accurate default, so a well-converged
        forward solve alongside a poorly converged adjoint one shows that the
        adjoint really ran through the solver it was given.
        """
        solver = self._fresh_solver()
        solver.set_adjoint_solver(
            lx.GMRES(rtol=1e-12, atol=1e-12, max_steps=1))

        _, adjoint_err = solver.adjoint_solve(self.V, self.J, self.I,
                                              self.b, self.x0)
        _, forward_err = solver.jax_solve(self.V, self.I, self.J,
                                          self.b, self.x0)

        assert float(forward_err) < 1e-8
        assert float(adjoint_err) > 1e-8

    def test_adjoint_solver_set_after_forward_solve(self):
        """Configuring the adjoint later still takes effect.

        The compiled solve bakes in its solver, so a single cached compilation
        would silently keep serving the forward solver here.
        """
        solver = self._fresh_solver()
        solver.jax_solve(self.V, self.I, self.J, self.b, self.x0)

        solver.set_adjoint_solver(
            lx.GMRES(rtol=1e-12, atol=1e-12, max_steps=1))
        _, adjoint_err = solver.adjoint_solve(self.V, self.J, self.I,
                                              self.b, self.x0)
        assert float(adjoint_err) > 1e-8


if __name__ == "__main__":
    unittest.main()
