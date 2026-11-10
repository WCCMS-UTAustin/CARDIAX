"""

    Tests the symmetrically Dirichlet-eliminated sparse operator.

    Builds a small system with a handful of Dirichlet dofs and checks that the
    stored/eliminated matrix (``operator.as_matrix()``):

      * has identity Dirichlet rows (row ``d`` is ``e_d``, i.e. ``(d, d) == 1``
        and off-diagonal entries in that row are 0);
      * has zeroed Dirichlet columns (no free<->Dirichlet coupling remains);
      * equals the raw tangent on the free-free block;
      * has no duplicate ``(i, j)`` indices;

    that the operator acts symmetrically and carries the positive-semidefinite
    tag a symmetric Krylov method (CG/MINRES) requires, and that
    ``operator.mv(x)`` matches ``operator.as_matrix() @ x``.

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
from cardiax.operators import eliminate_matrix_data, is_numerically_symmetric
from lineax._tags import positive_semidefinite_tag


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


def _mock_problem(init_guess, I, J, bc_inds):
    mock_problem = MagicMock(Problem)
    mock_problem.num_total_dofs_all_vars = len(init_guess)
    mock_problem.I = np.array(I)
    mock_problem.J = J
    mock_problem.bc_inds = onp.asarray(bc_inds, dtype=onp.int32)
    mock_problem.internal_vars = {'u': []}
    mock_problem.internal_vars_surfaces = {'u': []}
    return mock_problem


class Test_EliminatedOperator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        onp.random.seed(0)
        n = 8
        # Dense, fully-populated (every dof couples to every other), symmetric
        # positive-definite test matrix so every dof has a diagonal entry.
        B = onp.random.rand(n, n)
        A_np = B + B.T + n * onp.eye(n)
        cls.n = n
        cls.A_np = A_np
        cls.V, cls.I, cls.J = _get_VIJ_bcoo(A_np)
        cls.bc_inds = onp.array([1, 4], dtype=onp.int32)
        cls.free_inds = onp.array([d for d in range(n) if d not in cls.bc_inds])

        init_guess = onp.zeros(n)
        mock_problem = _mock_problem(init_guess, cls.I, cls.J, cls.bc_inds)
        cls.solver = Solver_Base(mock_problem, init_guess,
                                 precond='none', line_search_flag=False)
        # I/J are already unique here, so reduceV would be a pass-through
        # relabeling; populate the operator directly with the eliminated
        # reduction of the raw data, using the same static masks the solver
        # itself precomputed in __post_init__.
        cls.solver.A.matrix.data = eliminate_matrix_data(
            np.array(cls.V), cls.solver.elim_keep, cls.solver.elim_red_diag)
        cls.A_mat = cls.solver.A.as_matrix()
        # Densify (BCOO -> dense) for straightforward indexing/comparison.
        cls.A_dense = onp.asarray(cls.A_mat.todense())

    def test_dirichlet_rows_are_identity(self):
        """Each Dirichlet row d is e_d: (d, d) == 1, off-diagonal entries 0."""
        for d in self.bc_inds:
            row = self.A_dense[d, :]
            expected = onp.zeros(self.n)
            expected[d] = 1.0
            onptest.assert_allclose(row, expected, atol=1e-12)

    def test_dirichlet_columns_are_zeroed(self):
        """No free<->Dirichlet coupling remains in the Dirichlet columns."""
        for d in self.bc_inds:
            col = self.A_dense[:, d]
            for i in self.free_inds:
                self.assertAlmostEqual(float(col[i]), 0.0, places=12)

    def test_free_free_block_matches_raw_tangent(self):
        """The free-free block is unchanged from the raw assembled tangent."""
        free = onp.ix_(self.free_inds, self.free_inds)
        onptest.assert_allclose(self.A_dense[free], self.A_np[free], rtol=1e-10, atol=1e-12)

    def test_no_duplicate_indices(self):
        """The eliminated sparsity pattern has no duplicate (i, j) entries."""
        idx = onp.asarray(self.A_mat.indices)
        # Encode each (row, col) pair as a single integer key to detect dupes.
        keys = idx[:, 0].astype(onp.int64) * (self.n + 1) + idx[:, 1].astype(onp.int64)
        self.assertEqual(len(onp.unique(keys)), len(keys))

    def test_operator_is_psd_tagged(self):
        """The stored operator is tagged positive-semidefinite (CG-eligible)."""
        self.assertIn(positive_semidefinite_tag, self.solver.A.tags)

    def test_operator_acts_symmetrically(self):
        """<A x, y> == <x, A y>: eliminating rows and columns keeps symmetry.

        The raw tangent here is symmetric, so row-only elimination - which
        would leave the free-Dirichlet coupling in the columns - would break
        this.
        """
        key = jax.random.PRNGKey(1)
        self.assertTrue(is_numerically_symmetric(self.solver.A, key))

    def test_mv_matches_as_matrix(self):
        """operator.mv(x) matches operator.as_matrix() @ x for a random x."""
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (self.n,), dtype=np.float64)
        mv_result = onp.asarray(self.solver.A.mv(x))
        dense_result = self.A_dense @ onp.asarray(x)
        onptest.assert_allclose(mv_result, dense_result, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
