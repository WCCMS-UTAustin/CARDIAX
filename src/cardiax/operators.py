"""

custom linear operators that extend lineax linear operators!!

"""

import jax
import jax.numpy as np
import jax.experimental.sparse as js

import lineax as lx

class SparseMatrixLinearOperator(lx.MatrixLinearOperator):
    """Defines the matrix operator that will be used in the solver.
    The matrix object is given initially as a template for the object, and
    the dirichlet dofs are required because when solving Ax=b, we ignore
    the dofs that are fixed by Dirichlet boundary conditions.

    The stored matrix (``self.matrix``) is the **symmetrically Dirichlet-
    eliminated** tangent: both the Dirichlet rows and the Dirichlet columns are
    removed from the sparsity pattern (not zeroed in place), and the Dirichlet
    diagonal entries are set to 1 (identity rows). This is the operator that
    the Krylov solver iterates on, and it is also exactly what
    ``as_matrix()`` (inherited from ``lx.MatrixLinearOperator``) returns, so a
    matrix-consuming preconditioner builds its hierarchy on the same operator
    the solver uses.

    The free-Dirichlet coupling that symmetric elimination drops from the
    operator does not vanish: it is applied to the fixed Dirichlet increment
    and moved onto the right-hand side of the linear solve (the boundary-load
    "lift" term), which happens outside this class.

    IMPORTANT: Once a discretization is chosen, the sparsity of the operator
    is fixed. Thus, once the operator is created, the data is the only thing that
    is updated in solver.

    Args:
        matrix (jax.experimental.sparse.BCOO): a batched COO matrix implemented in JAX,
            which is a sparse matrix with indices and data, already reduced to the
            eliminated (Dirichlet row/column dropped) sparsity pattern. For use in
            solver, this will be initialized in a specific way.
        dirichlet_dofs (np.array): the indices of the Dirichlet boundary conditions.

    """
    dirichlet_dofs: np.array

    def __init__(self, matrix, dirichlet_dofs, tags=None):
        if tags is not None:
            super().__init__(matrix, tags)
        else:
            super().__init__(matrix)
        self.dirichlet_dofs = dirichlet_dofs
        # raise an exception if the dirichlet dofs are not integers
        # print(np.issubdtype(x.dtype, np.floating)) # Output: True
        if not np.issubdtype(dirichlet_dofs.dtype, np.integer):
            raise Exception("dirichlet_dofs must be a np.array of INTEGERS.")

        return

    def mv(self, dofs):
        """Plain sparse matrix-vector product with the stored (already
        Dirichlet-eliminated) matrix. The sparsify function allows BCOO
        objects as input and removes the 0 computations that are standard
        with JAX. No per-call row/column masking is needed here: the
        elimination is baked into ``self.matrix`` at construction/write time.

        Args:
            dofs (np.array): vector describing current state of the system

        Returns:
            res_vec (np.array): the result of the matrix-vector multiplication,
            if the data is the V matrix, then this will be the residual vector.
        """
        return js.sparsify(lambda m, v: m @ v)(self.matrix, dofs)

    # issue with transpose: the mv method is overwritten when a linear opeartor is tranposed. idk if this is
    # an issue with lineax, or an issue with how we implemented sparsematrixlinearoperator...
    def transpose(self):
        # The symmetrically eliminated matrix is its own transpose.
        return self


def compute_elimination_masks(I: np.array, J: np.array, bc_inds: np.array, n_dofs: int):
    """Precompute the static masks used to reduce a raw (deduplicated) sparse
    pattern to the symmetrically Dirichlet-eliminated pattern.

    Given the deduplicated sparsity pattern ``(I, J)`` of the assembled
    tangent and the set of Dirichlet dofs, this returns:

    - ``keep``: a boolean mask selecting entries that survive elimination -
      the free-free block, plus the Dirichlet diagonal entries (which are
      reused as identity rows rather than dropped).
    - ``red_diag``: a boolean mask, aligned to the *reduced* (``keep``-masked)
      arrays, marking which surviving entries are Dirichlet diagonals (so
      their value can be overwritten to 1).

    Both masks are static functions of the fixed sparsity pattern and the
    fixed set of Dirichlet dofs, so they are computed once and reused for
    every matrix write (forward and adjoint). Because ``keep``/``red_diag``
    are symmetric under swapping ``I`` and ``J``, the same masks serve the
    transposed pattern used by the adjoint solve.

    Args:
        I (np.array): row indices of the deduplicated sparsity pattern.
        J (np.array): column indices of the deduplicated sparsity pattern.
        bc_inds (np.array): indices of the Dirichlet dofs.
        n_dofs (int): total number of degrees of freedom.

    Returns:
        keep (np.array[bool]): selection mask into the raw pattern.
        red_diag (np.array[bool]): Dirichlet-diagonal mask into the reduced
            (``keep``-masked) pattern.
        I_red, J_red (np.array): the reduced row/column indices, i.e.
            ``I[keep]``, ``J[keep]``.
    """
    is_d = np.zeros(n_dofs, dtype=bool)
    is_d = is_d.at[bc_inds].set(True, mode="drop")

    is_d_I = is_d[I]
    is_d_J = is_d[J]
    keep = (~is_d_I & ~is_d_J) | (is_d_I & (I == J))

    I_red, J_red = I[keep], J[keep]
    red_diag = is_d[I_red] & (I_red == J_red)

    return keep, red_diag, I_red, J_red


def assert_dirichlet_diagonals_present(I: np.array, J: np.array, bc_inds: np.array, n_dofs: int) -> None:
    """Assert that every Dirichlet dof already has a diagonal entry in the
    deduplicated sparsity pattern ``(I, J)``.

    Symmetric elimination reuses the existing ``(d, d)`` entry as the
    Dirichlet identity row rather than inserting a new one, so no duplicate
    indices are ever created. That requires the diagonal entry to already be
    present for every Dirichlet dof. Standard finite-element assembly always
    produces it (every dof couples to itself), so this should never fail in
    practice; if it ever does, the fix is to inject the missing ``(d, d)``
    entry into the assembled pattern *before* deduplication so it merges in
    cleanly, rather than adding it here at solve time.

    Args:
        I (np.array): row indices of the deduplicated sparsity pattern.
        J (np.array): column indices of the deduplicated sparsity pattern.
        bc_inds (np.array): indices of the Dirichlet dofs.
        n_dofs (int): total number of degrees of freedom.

    Raises:
        AssertionError: if some Dirichlet dof has no ``(d, d)`` entry.
    """
    import numpy as onp

    bc_inds = onp.asarray(bc_inds)
    bc_inds = bc_inds[bc_inds < n_dofs]
    diag_mask = onp.asarray(I) == onp.asarray(J)
    diag_dofs = set(onp.asarray(I)[diag_mask].tolist())
    missing = [int(d) for d in bc_inds.tolist() if int(d) not in diag_dofs]
    assert not missing, (
        "Dirichlet dof(s) missing a diagonal entry in the assembled sparsity "
        f"pattern, so there is no (d, d) entry to reuse for elimination: {missing}"
    )


def eliminate_matrix_data(V: np.array, keep: np.array, red_diag: np.array) -> np.array:
    """Reduce raw matrix data to the symmetrically Dirichlet-eliminated data,
    given the static ``keep``/``red_diag`` masks from
    :func:`compute_elimination_masks`.

    Args:
        V (np.array): raw matrix data, aligned to the full (pre-elimination)
            sparsity pattern.
        keep (np.array[bool]): selection mask into ``V``.
        red_diag (np.array[bool]): Dirichlet-diagonal mask into the reduced
            (``keep``-masked) data.

    Returns:
        np.array: the eliminated matrix data, aligned to the reduced pattern
        (``I[keep]``, ``J[keep]``), with Dirichlet diagonal entries set to 1.
    """
    data = V[keep]
    data = np.where(red_diag, 1.0, data)
    return data


def is_numerically_symmetric(operator, key, n_probes: int = 4, atol: float = 1e-8,
                             rtol: float = 1e-6) -> bool:
    """Spot-check whether ``operator`` acts symmetrically: ``<A x, y> == <x, A y>``.

    Probes the operator with ``n_probes`` random vector pairs and compares the
    two bilinear forms. Intended for tests / one-off verification before trusting
    a symmetric Krylov method (CG/MINRES) on the operator - not a per-solve
    runtime check.

    Args:
        operator: a lineax linear operator exposing ``.mv`` and ``.in_size()``.
        key: a ``jax.random.PRNGKey`` for the random probes.
        n_probes (int): number of random ``(x, y)`` pairs to test.
        atol (float), rtol (float): tolerances for the symmetric comparison.

    Returns:
        bool: ``True`` if every probe satisfies
        ``|<A x, y> - <x, A y>| <= atol + rtol * |<A x, y>|``.
    """
    n = operator.in_size()
    ok = True
    for _ in range(n_probes):
        key, kx, ky = jax.random.split(key, 3)
        x = jax.random.normal(kx, (n,), dtype=np.float64)
        y = jax.random.normal(ky, (n,), dtype=np.float64)
        lhs = np.vdot(operator.mv(x), y)
        rhs = np.vdot(x, operator.mv(y))
        ok = ok and bool(np.abs(lhs - rhs) <= atol + rtol * np.abs(lhs))
    return ok
