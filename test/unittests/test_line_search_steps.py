"""

    Tests the configurable Newton line-search damping grid.

    Covers:
      * the default grid is the 20-point geometric one, ending at the full step;
      * :func:`geometric_line_search_steps` honors its resolution and ratio,
        and rejects nonsensical ones;
      * a caller-supplied grid is used verbatim, and the line search picks the
        step from it that minimizes the residual norm;
      * the search never returns a step outside the grid it was given;
      * malformed grids are rejected at construction time.

"""

import numpy as onp
import jax
import jax.numpy as np
jax.config.update("jax_enable_x64", True)

import unittest
from unittest.mock import MagicMock
import numpy.testing as onptest

from cardiax._solver import (
    Solver_Base,
    geometric_line_search_steps,
    DEFAULT_LINE_SEARCH_STEPS,
)
from cardiax import Problem


def _mock_problem(n):
    """A problem with a trivial diagonal sparsity pattern and no Dirichlet dofs."""
    diag = onp.arange(n, dtype=onp.int32)
    mock_problem = MagicMock(Problem)
    mock_problem.num_total_dofs_all_vars = n
    mock_problem.I = np.array(diag)
    mock_problem.J = np.array(diag)
    mock_problem.bc_inds = onp.array([], dtype=onp.int32)
    mock_problem.internal_vars = {'u': []}
    mock_problem.internal_vars_surfaces = {'u': []}
    return mock_problem


class Test_LineSearchSteps(unittest.TestCase):

    N = 4

    def _solver(self, **kwargs):
        init_guess = onp.zeros(self.N)
        return Solver_Base(_mock_problem(self.N), init_guess, precond='none',
                           **kwargs)

    def _search(self, solver, best_alpha):
        """Run the line search against a residual minimized at ``best_alpha``."""
        # line_search calls res_norm_part(dofs, inc, alpha) under vmap; the
        # Newton solver installs it, so stand in for it here.
        solver.res_norm_part = lambda dofs, inc, alpha: (alpha - best_alpha) ** 2
        dofs = np.zeros(self.N)
        inc = np.ones(self.N)
        # dofs is zero and inc is all ones, so every entry of the result is the
        # chosen damping fraction.
        return float(solver.line_search(dofs, inc)[0])

    # ----------------------------------------------------------------- #
    def test_default_grid_is_geometric(self):
        """With no grid supplied the solver uses the geometric default."""
        onptest.assert_allclose(
            onp.asarray(self._solver().line_search_steps),
            0.5 ** onp.arange(DEFAULT_LINE_SEARCH_STEPS - 1, -1, -1))

    def test_default_grid_reaches_the_full_step(self):
        """The coarsest sampled fraction is the undamped Newton step."""
        self.assertEqual(float(self._solver().line_search_steps[-1]), 1.0)

    def test_geometric_grid_honors_resolution_and_ratio(self):
        """num_steps sets the length, ratio sets the spacing."""
        steps = onp.asarray(geometric_line_search_steps(num_steps=5, ratio=0.1))
        onptest.assert_allclose(steps, [1e-4, 1e-3, 1e-2, 1e-1, 1.0], rtol=1e-12)

    def test_geometric_grid_rejects_bad_arguments(self):
        """A grid needs at least one point and a ratio strictly inside (0, 1)."""
        with self.assertRaises(ValueError):
            geometric_line_search_steps(num_steps=0)
        for ratio in (0.0, 1.0, -0.5):
            with self.assertRaises(ValueError):
                geometric_line_search_steps(ratio=ratio)

    def test_custom_grid_is_used_verbatim(self):
        """A supplied grid is stored as given, not merged with the default."""
        custom = [0.05, 0.3, 0.85]
        onptest.assert_allclose(
            onp.asarray(self._solver(line_search_steps=custom).line_search_steps),
            custom)

    def test_line_search_picks_the_best_custom_step(self):
        """The step minimizing the residual is chosen out of the custom grid."""
        solver = self._solver(line_search_steps=[0.05, 0.3, 0.85])
        self.assertAlmostEqual(self._search(solver, best_alpha=0.32), 0.3)
        self.assertAlmostEqual(self._search(solver, best_alpha=0.9), 0.85)

    def test_line_search_picks_the_best_default_step(self):
        """The default geometric grid resolves small feasible steps.

        A uniformly spaced 20-point grid bottoms out at 0.05, so a residual
        minimized far below that would be answered with 0.05; the geometric
        grid reaches 2**-19 instead.
        """
        chosen = self._search(self._solver(), best_alpha=2.0 ** -15)
        self.assertAlmostEqual(chosen, 2.0 ** -15)

    def test_single_step_grid_forces_that_step(self):
        """A one-element grid disables the search without disabling damping."""
        solver = self._solver(line_search_steps=[0.25])
        self.assertAlmostEqual(self._search(solver, best_alpha=1.0), 0.25)

    def test_malformed_grids_are_rejected(self):
        """An empty or multi-dimensional grid is a construction-time error."""
        for bad in ([], [[0.5, 1.0], [0.25, 0.75]]):
            with self.assertRaises(ValueError):
                self._solver(line_search_steps=bad)


if __name__ == "__main__":
    unittest.main()
