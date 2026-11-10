from typing import Callable
import jax
import jax.numpy as np
import jax.flatten_util
import jax.experimental.sparse as js
from abc import ABC
from dataclasses import dataclass, field
from cardiax import Problem
import functools as fctls
from cardiax.common import timeit
from jaxtyping import ArrayLike

# lineax imports
from cardiax.operators import (
    SparseMatrixLinearOperator,
    compute_elimination_masks,
    eliminate_matrix_data,
    assert_dirichlet_diagonals_present,
)
import lineax as lx

# preconditioners
from cardiax.preconditioners._preconditioners import (
    _set_preconditioning_method
)

# logging / debugging
from cardiax import logger


#: Number of damping fractions in the default line-search grid.
DEFAULT_LINE_SEARCH_STEPS = 20


def geometric_line_search_steps(num_steps: int = DEFAULT_LINE_SEARCH_STEPS,
                                ratio: float = 0.5) -> jax.Array:
    """Build a geometric grid of line-search damping fractions.

    Returns ``[ratio**(num_steps-1), ..., ratio**2, ratio, 1]``: finest near 0
    and coarsest at the full step. This is the default the solver uses, and it
    is exported so that callers who want the same shape of grid at a different
    resolution can ask for one instead of writing the expression out.

    A uniformly spaced grid is a poor default here. With 20 uniform points the
    smallest fraction is 0.05, and when the Jacobian is near-singular the Newton
    increment is large enough that even 0.05 of it inverts elements: the whole
    line evaluates to NaN, and the ``nanargmin`` fallback then takes the full
    step, which is the worst available choice. Geometric spacing reaches
    ``ratio**(num_steps-1)`` with the same number of residual evaluations and
    resolves the tiny feasible step instead.

    Args:
        num_steps (int): number of damping fractions to sample; must be >= 1.
        ratio (float): common ratio between consecutive fractions, in (0, 1).

    Returns:
        np.array: the damping fractions, ascending, ending at 1.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must lie in (0, 1), got {ratio}.")
    return ratio ** np.arange(num_steps - 1, -1, -1, dtype=np.float64)


@dataclass
class Solver_Base(ABC):
    """Defines the base class for solver to be used for other specified solvers
    such as Newton, dynamic_relax, etc. It populates all the necessary functions
    that should be required in more custom solvers such as handling boundary condition
    data, creating the residual vector, line searches, preconditioners (may want to move),
    matrix system solvers, and adjoint capabilities.

    Args:
        problem (Problem): the problem object that is being solved
        initial_guess (np.array): the initial guess for the solver
        precond (bool): whether to use a preconditioner in the solver
        line_search_flag (bool): whether to use a line search in the solver
        line_search_steps (array-like): the damping fractions the line search
            samples, as any 1-D array-like. Defaults to the geometric grid from
            :func:`geometric_line_search_steps`. Only consulted when
            ``line_search_flag`` is set.
        use_petsc (bool): whether to use the PETSc solver (remove?)
        petsc_options (list): options for the PETSc solver (remove?)
    """

    problem: Problem
    initial_guess: np.array
    precond: str = field(default='none')
    line_search_flag: bool = field(default=True)
    line_search_steps: ArrayLike = field(default=None)
    use_petsc: bool = field(default=False)
    petsc_options: list = field(default_factory=list) #Not sure what should be here
    
    def __post_init__(self):

        ### use a few assertions to force the user to use a correctly sized initial guess
        if not (self.problem.num_total_dofs_all_vars == len(self.initial_guess.reshape(-1))):
            raise Exception("The initial guess you supplied has the wrong shape. Initial guess should" \
            " have a length of self.problem.num_total_dofs_all_vars.")
        
        indices = np.vstack([self.problem.I, self.problem.J]).T
        unique_indices, mask = np.unique(indices, axis=0, return_inverse=True)
        self.mask = mask.flatten()
        self.I, self.J = unique_indices[:, 0], unique_indices[:, 1]

        temp_BCOO = js.BCOO((np.arange(len(self.I)), np.vstack([self.I, self.J]).T),
                            shape=(self.problem.num_total_dofs_all_vars, self.problem.num_total_dofs_all_vars)).sort_indices()
        self.V_mask = temp_BCOO.data
        self.I, self.J = temp_BCOO.indices[:, 0], temp_BCOO.indices[:, 1]

        # TODO: update the dtype for this structure.
        temp_BCOO = temp_BCOO.astype(np.float64)

        # Static masks that reduce the raw (deduplicated) sparsity pattern
        # (self.I, self.J) to the symmetrically Dirichlet-eliminated pattern:
        # the free-free block, plus the Dirichlet diagonal entries (reused as
        # identity rows). These depend only on the fixed pattern and the fixed
        # set of Dirichlet dofs, so they are computed once here and reused for
        # every per-solve matrix write (forward and adjoint alike, since the
        # masks are symmetric under swapping I and J).
        assert_dirichlet_diagonals_present(
            self.I, self.J, self.problem.bc_inds, self.problem.num_total_dofs_all_vars)
        self.elim_keep, self.elim_red_diag, self.I_red, self.J_red = compute_elimination_masks(
            self.I, self.J, self.problem.bc_inds, self.problem.num_total_dofs_all_vars)

        temp_BCOO_elim = js.BCOO((eliminate_matrix_data(temp_BCOO.data, self.elim_keep, self.elim_red_diag),
                                  np.vstack([self.I_red, self.J_red]).T),
                                 shape=(self.problem.num_total_dofs_all_vars,
                                        self.problem.num_total_dofs_all_vars)).sort_indices()
        # Tag the operator positive-semidefinite so symmetric Krylov methods
        # (CG/MINRES) may be used on it. The tag is a hint only - GMRES/BiCGStab
        # ignore it - so this does not change the behavior of solvers that do
        # not exploit symmetry.
        self.A = SparseMatrixLinearOperator(
            temp_BCOO_elim, self.problem.bc_inds, tags=lx.positive_semidefinite_tag)

        self.reduceV = jax.jit(self.reduceV)

        # Materialize the line-search grid once. Storing it as an array (rather
        # than rebuilding it per Newton step) keeps it a compile-time constant
        # of the vmapped residual evaluation, so a caller-supplied grid costs no
        # more than the default one.
        self.line_search_steps = self._validate_line_search_steps(
            self.line_search_steps)

        # self.line_search = jax.jit(self.line_search)

        self.solver = lx.BiCGStab(rtol=1e-10, atol=1e-10, max_steps=100000)
        # Optional dedicated solver for the adjoint linear solve. None means
        # the adjoint reuses self.solver, which is the historical behavior.
        self.adjoint_solver = None
        # One compiled solve per distinct solver object, built on first use.
        # The solver has to be baked in at trace time, so a single compiled
        # solve cannot serve two different solvers; keying the cache on the
        # solver instead of compiling once in the constructor is what lets the
        # forward and adjoint solvers differ, and lets either be swapped after
        # construction (which callers routinely do) without a stale compile.
        self._compiled_solves = []

        # define the preconditioning method of choice
        # for now, we have true/false options, but this can be expanded upon
        # by updating the method below
        #
        # passing A can provide helpful info about the structure of the system we are
        # preconditioning to allow for some data structures to be defined.
        self.get_preconditioner = _set_preconditioning_method(self.precond, self.A)

        """
        TODO: We need to switch kernel creation to track function signature for where
        specific variables go. Currently, the ordering of dictionaries can be changed
        in JAX functions which is mainly happening in the line_search function.
        """
        # Ordering for internal vars, hacky for now. 
        # Should use signature in the future to match up the order
        # a priori after the user defines the kernel
        key = list(self.problem.internal_vars.keys())[0]
        self.order_internal_vars = {key: list(self.problem.internal_vars[key])}
        self.order_internal_vars_surfaces = {}
        temp = {}
        for bc in self.problem.internal_vars_surfaces[key]:
            temp[bc] = list(self.problem.internal_vars_surfaces[key][bc])
        self.order_internal_vars_surfaces[key] = temp

    # This function has the potential to be removed in the future
    # if sparse gets out of experimental and repeated indices are 
    # handled better
    def reduceV(self, V: ArrayLike) -> ArrayLike:
        """Reduces the V matrix to only the unique indices.
        Current versions of JAX, adding repeated indices is significantly
        slower than performing an initial reduction.

        Args:
            V (np.array): Accepts the V matrix obtained from 
            newton_update_helper

        Returns:
            vec (np.array): shortened array of V as large as unique indices
        """
        temp = np.zeros_like(self.I, dtype=np.float64)
        temp = temp.at[self.mask].add(V)
        return temp[self.V_mask]

    # Used to set values of res_vec array
    def apply_bc_vec(self, res_vec: ArrayLike, dofs: ArrayLike) -> jax.Array:
        """Applys boundary conditions to the residual vector in place.

        Args:
            res_vec (np.array): The residual vector to modify.
            dofs (np.array): The degrees of freedom vector.

        Returns:
            np.array: The modified residual vector.
        """

        dirichlet_dofs, dirichlet_vals = self.problem.get_boundary_data()

        res_vec = res_vec.at[dirichlet_dofs].set(dofs[dirichlet_dofs])
        res_vec = res_vec.at[dirichlet_dofs].add(-dirichlet_vals, unique_indices=True)

        return res_vec
    
    def apply_bc(self, res_fn: Callable) -> Callable:
        """Used to set values of the residual function. Only
        done in the adjoint, so may be able to change

        Args:
            res_fn (callable): computes the residual vector
        """

        def A_fn(dofs):
            """Apply Dirichlet boundary conditions
            """
            res_vec = res_fn(dofs)
            return self.apply_bc_vec(res_vec, dofs)

        return A_fn

    def assign_bc(self, dofs: ArrayLike) -> jax.Array:
        """Assign Dirichlet boundary conditions to the degrees of freedom.

        Args:
            dofs (np.array): Degrees of freedom vector to modify.

        Returns:
            np.array: The modified degrees of freedom vector.
        """
        # Assigns bc vals to dofs
        dirichlet_dofs, dirichlet_vals = self.problem.get_boundary_data()
        dofs = dofs.at[dirichlet_dofs].set(dirichlet_vals)
        return dofs

    # NOTE: where is/was this getting used?
    def assign_ones_bc(self, dofs: ArrayLike) -> jax.Array:
        """Assigns ones to the Dirichlet boundary conditions.
        NOT USED

        Args:
            dofs (np.array): Degrees of freedom vector to modify.

        Returns:
            np.array: The modified degrees of freedom vector.
        """
        # Assigns ones to bc
        #TODO: Only used in jacobi to neglect preconditioning BC dofs
        # Can remove if making preconditioner scripts
        dirichlet_dofs, dirichlet_vals = self.problem.get_boundary_data()
        dofs = dofs.at[dirichlet_dofs].set(1.)
        return dofs


    def assign_zeros_bc(self, dofs: ArrayLike) -> jax.Array:
        """Assign zeros to BCs.
        NOT USED

        Args:
            dofs (np.array): Degrees of freedom vector to modify.

        Returns:
            np.array: The modified degrees of freedom vector.
        """
        # No longer necessary since zeroing occurs in the .mv method
        # of the sparse matrix object
        #TODO: Double check this won't be needed before removing
        dirichlet_dofs, dirichlet_vals = self.problem.get_boundary_data()
        dofs = dofs.at[dirichlet_dofs].set(0.)
        return dofs

    def copy_bc(self, dofs: ArrayLike) -> jax.Array:
        """Copy the Dirichlet boundary conditions from the dofs

        Args:
            dofs (np.array): Degrees of freedom vector to modify.

        Returns:
            np.array: The modified degrees of freedom vector.
        """
        # Copies current bcs to new array
        # Used for the lifting of Dirichlet BCs
        #TODO: Check if this is necessary or can be done using assign_bc
        dirichlet_dofs, _ = self.problem.get_boundary_data()
        new_dofs = np.zeros_like(dofs)
        new_dofs = new_dofs.at[dirichlet_dofs].set(dofs[dirichlet_dofs])
        return new_dofs

    def res_norm_fn(self, dofs: jax.Array, inc: jax.Array, alpha: float, internal_vars: dict, internal_vars_surfaces: dict) -> float:
        """Compute the residual norm for the line search method.
        We leave dependence on internal_vars and internal_vars_surfaces,
        so we don't have to jit compile each time they update.

        Args:
            dofs (np.array): DOF array (base from the solver)
            inc (np.array): DOF increment array (direction to move dofs)
            alpha (float): scalar that determines how far in the incremented direction to move
            internal_vars (list): list with the internal variables used in the PDE
            internal_vars_surfaces (list): list of the internal variables on the surface used in the PDE

        Returns:
            vec (np.array): the norm of the residual vector
        """

        """
        TODO: Once we switch to tracking variable locations with function signatures,
        we can get rid of this reordering step. This is needed because dictionary keys
        aren't preserved over vmapping
        """
        res_vec = self.problem.compute_residual_helper(dofs + alpha*inc, internal_vars, internal_vars_surfaces)
        res_vec = self.apply_bc_vec(res_vec, dofs + alpha*inc)
        return np.linalg.norm(res_vec)

    @staticmethod
    def _validate_line_search_steps(steps: ArrayLike) -> jax.Array:
        """Normalize the ``line_search_steps`` argument into a 1-D array.

        Args:
            steps (array-like): the damping fractions to sample, or ``None`` to
                take the default geometric grid.

        Returns:
            np.array: the damping fractions as a float array.

        Raises:
            ValueError: if ``steps`` is not a non-empty 1-D array-like.
        """
        if steps is None:
            return geometric_line_search_steps()
        steps = np.asarray(steps, dtype=np.float64)
        if steps.ndim != 1 or steps.size == 0:
            raise ValueError(
                "line_search_steps must be a non-empty 1-D array of damping "
                f"fractions; got shape {steps.shape}.")
        return steps

    def line_search(self, dofs: ArrayLike, inc: ArrayLike) -> jax.Array:
        """Damp the Newton increment by the sampled step that minimizes the
        residual norm.

        Every damping fraction in ``self.line_search_steps`` is evaluated (in
        parallel, via ``vmap``) and the one giving the smallest residual norm is
        taken. See :func:`geometric_line_search_steps` for the default grid and
        the ``line_search_steps`` constructor argument for supplying your own.

        Args:
            dofs (np.array): current DOF array
            inc (np.array): the Newton increment to damp

        Returns:
            np.array: ``dofs`` advanced by the best sampled fraction of ``inc``
        """
        steps = self.line_search_steps
        vres = jax.vmap(self.res_norm_part, in_axes=(None, None, 0))
        res_vals = vres(dofs, inc, steps)
        min_ind = np.nanargmin(res_vals)
        # nanargmin returns [-1] if all the values are NaN, which in
        # this case is the desired behavior as it will use the full step
        # and surface the NaN that couldn't be resolved by the line
        # search

        return dofs + steps[min_ind] * inc

    def set_adjoint_solver(self, solver) -> None:
        """Choose the linear solver used for the adjoint solve.

        The forward and adjoint linear systems are solved with the same
        operator but are not equally hard: the forward solve runs at states
        far from equilibrium, where the tangent may be poorly conditioned or
        indefinite, while the adjoint is evaluated at the converged state.
        A method that is robust enough for the forward iterates is often
        needlessly slow for the adjoint, and vice versa, so the two are
        allowed to differ.

        Args:
            solver: a lineax solver instance to use for the adjoint solve, or
                ``None`` to go back to reusing the forward solver
                (``self.solver``), which is the default.
        """
        self.adjoint_solver = solver

    def _compiled_solve_for(self, solver):
        """Return the compiled linear solve bound to ``solver``.

        The solver is baked in when the solve is traced, so one compiled solve
        cannot serve two solvers. Rather than compile once in the constructor,
        which would freeze whichever solver happened to be installed then,
        each distinct solver gets its own compiled solve on first use. The list
        holds at most one entry per solver in play (the forward one and, if
        configured, the adjoint one), and identity comparison is used so that
        reinstalling the same solver object reuses its compiled solve.
        """
        for known, compiled in self._compiled_solves:
            if known is solver:
                return compiled
        compiled = timeit(jax.jit(
            fctls.partial(self._linear_solve_with_operator, solver)))
        self._compiled_solves.append((solver, compiled))
        return compiled

    def _linear_solve_with_operator(self, solver, mat_data: ArrayLike,
                                    row_inds: ArrayLike, col_inds: ArrayLike,
                                    b: ArrayLike, x0: ArrayLike) -> tuple:
        """Solve ``Ax = b`` with ``solver``, populating ``self.A`` from the data.

        Shared body of the forward and adjoint solves; the two differ only in
        which solver they pass and, for the adjoint, in passing the transposed
        sparsity pattern.

        ``row_inds``/``col_inds`` are the *raw* (un-eliminated) pattern, aligned
        with ``mat_data``. They are reduced to the symmetrically Dirichlet-
        eliminated pattern that ``self.A`` stores, using the static masks
        precomputed in ``__post_init__``; those masks are symmetric under
        swapping rows and columns, so they reduce the adjoint's transposed
        pattern just as well as the forward one.

        Args:
            solver: the lineax solver to run.
            mat_data (np.array): data array for the BCOO
            row_inds (np.array): row indices used for the action of A
            col_inds (np.array): column indices used for the action of A
            b (np.array): right hand side of the linear system
            x0 (np.array): initial guess for the solver

        Returns:
            x (np.array): solution to the linear system
            err (float): the residual norm of the solution
        """
        A_fn = self.A
        A_fn.matrix.data = eliminate_matrix_data(mat_data, self.elim_keep,
                                                 self.elim_red_diag)
        A_fn.matrix.indices = np.vstack([row_inds[self.elim_keep],
                                         col_inds[self.elim_keep]]).T
        pc = self.get_preconditioner(A_fn.matrix)
        solution = lx.linear_solve(A_fn, b, solver,
                            options={"y0": x0, "preconditioner": pc}, throw=False)

        x = solution.value

        # likely want to avoid re-sizing from within the solver,
        # but err is not correct otherwise

        # Verify convergence
        err = np.linalg.norm(A_fn.mv(x) - b)

        return x, err

    def jax_solve(self, mat_data: ArrayLike, row_inds: ArrayLike,
                  col_inds: ArrayLike, b: ArrayLike,
                  x0: ArrayLike) -> tuple:
        """Solves the forward linear system Ax=b using the
        lineax solver with the defined method. This defaults
        to the BiCGstab solver, but can be changed with the
        solver.solver object.

        Args:
            V (np.array): data matrix for the BCOO
            I (np.array): Row indices used for action of A
            J (np.array): Column indices used for action of A
            b (np.array): right hand side of the linear system (-res_vec)
            x0 (np.array): initial guess for the solver

        Returns:
            #TODO: lineax solver returns a dictionary
            that gives info about the solver,
            and we may want to return this as well
            x (np.array): solution to the linear system
            err (float): the error of the solution
        """
        return self._compiled_solve_for(self.solver)(mat_data, row_inds,
                                                     col_inds, b, x0)

    def _raw_matvec(self, mat_data: ArrayLike, row_inds: ArrayLike,
                    col_inds: ArrayLike, x: ArrayLike) -> jax.Array:
        """Matrix-vector product with the raw (BC-free) tangent.

        Multiplies by the assembled tangent with **no** Dirichlet elimination on
        either the rows or the columns. Used to move the Dirichlet-column lift
        onto the right-hand side: applying the raw tangent to the fixed Dirichlet
        increment recovers exactly the column contribution that symmetric
        elimination discards from the stored operator.

        Args:
            mat_data (np.array): data matrix for the BCOO
            row_inds (np.array): row indices used for action of the tangent
            col_inds (np.array): column indices used for action of the tangent
            x (np.array): vector to multiply

        Returns:
            np.array: the raw (unconstrained) matrix-vector product
        """
        n = self.problem.num_total_dofs_all_vars
        raw = js.BCOO((mat_data, np.vstack([row_inds, col_inds]).T), shape=(n, n))
        return js.sparsify(lambda m, v: m @ v)(raw, x)

    def adjoint_solve(self, mat_data: ArrayLike, row_inds: ArrayLike,
                      col_inds: ArrayLike, b: ArrayLike,
                      x0: ArrayLike) -> tuple:
        """Solve the adjoint linear system, using the adjoint solver if set.

        Falls back to ``self.solver`` when no dedicated adjoint solver has
        been configured, so the default behavior is identical to solving the
        adjoint through :meth:`jax_solve`. See :meth:`set_adjoint_solver`.

        #NOTE: The indices are given as input, and the adjoint system is the
        transpose of the forward one, so callers pass the row and column
        indices swapped relative to :meth:`jax_solve`. Since the stored
        operator's Dirichlet elimination is symmetric, the operator is its own
        transpose, and populating it with the transposed pattern realizes
        exactly the transpose of the forward constrained tangent.

        Args:
            mat_data (np.array): data matrix for the BCOO
            row_inds (np.array): Row indices used for action of A
            col_inds (np.array): Column indices used for action of A
            b (np.array): right hand side of the linear system
            x0 (np.array): initial guess for the solver

        Returns:
            x (np.array): solution to the adjoint linear system
            err (float): the error of the solution
        """
        solver = self.adjoint_solver if self.adjoint_solver is not None \
            else self.solver
        return self._compiled_solve_for(solver)(mat_data, row_inds,
                                                col_inds, b, x0)

    def row_elimination(self, fn: Callable) -> Callable:
        """Add documentation

        Args:
            degree (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Never used
        #TODO: Probably should remove
        def fn_dofs_row(dofs):
            res_vec = fn(dofs)
            dirichlet_dofs, dirichlet_vals = self.problem.get_boundary_data()
            res_vec = res_vec.at[dirichlet_dofs].set(dofs[dirichlet_dofs], unique_indices = True)
            #res_vec = res_vec.at[dirichlet_dofs].add(-dirichlet_vals)

            return res_vec

        return fn_dofs_row

    def get_flatten_fn(self, fn_sol_list: Callable) -> Callable:
        """Flattens the output of the given function.
        Args:
            degree (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Used in adjoint
        def fn_dofs(dofs):
            val_list = fn_sol_list(dofs)
            return jax.flatten_util.ravel_pytree(val_list)[0]

        return fn_dofs

    def linear_incremental_solver(self, res_vec: ArrayLike, V: ArrayLike, dofs: ArrayLike) -> jax.Array:
        """Incremental solver that solves the linear system
        from the res_vec and V matrix. This is where
        line search can be toggled on or off
        """
        logger.debug("Solving linear system with lift solver...")
        b = -res_vec

        # x0 will always be correct at boundary locations
        x0_1 = self.assign_bc(np.zeros_like(b))
        x0_2 = self.copy_bc(dofs)
        x0 = x0_1 - x0_2

        # Symmetric elimination drops the Dirichlet columns of the stored
        # tangent, so the inhomogeneous Dirichlet lift no longer enters the
        # interior equations implicitly through those columns. Move it
        # explicitly onto the right-hand side: subtract the raw (BC-free)
        # tangent applied to the fixed Dirichlet increment x0, then restore the
        # target increment on the Dirichlet rows (which the eliminated operator
        # treats as the identity).
        dirichlet_dofs, _ = self.problem.get_boundary_data()
        b = b - self._raw_matvec(V, self.I, self.J, x0)
        b = b.at[dirichlet_dofs].set(x0[dirichlet_dofs])

        inc, err = self.jax_solve(V, self.I, self.J, b, x0)

        # FOR TESTING PURPOSES ONLY:
        # if the step fails, run it again...
        if jax.numpy.isnan(err):
            inc, err = self.jax_solve(V, self.I, self.J, b, x0)
            logger.debug(f"RERUN!!!!! JAX scipy linear solve res = {err}")
            # maybe update something global or print some easy-to-read
            # error code to the screen so we can parse results to find
            # the number of times this occurs...

        # logger.debug(f"result = {solution.result}")
        # logger.debug(f"num_steps = {solution.stats['num_steps']}")
        
        logger.debug(f"JAX scipy linear solve res = {err}")

        if self.line_search_flag:
            dofs = self.line_search(dofs, inc)
        else:
            dofs = dofs + inc

        return dofs

    def implicit_vjp(self, sol: ArrayLike, params, v: ArrayLike, use_petsc_adjoint: bool, petsc_options_adjoint: dict) -> ArrayLike:
        """Defines the implicit adjoint method for computing gradients. 
        TODO: This was manipulated to work with lineax, but could possibly 
        be improved since many old functions are only used here.
        TODO: Improve documentation after possible rework
        Args:
            sol (_type_): _description_
            params (_type_): _description_
            v (_type_): _description_
            use_petsc_adjoint (_type_): #TODO: Remove Petsc?
            petsc_options_adjoint (_type_): #TODO: Remove Petsc?
        """

        def constraint_fn(dofs, params):
            """c(u, p)
            """
            self.problem.set_params(params)
            res_fn = self.problem.compute_residual
            res_fn = self.get_flatten_fn(res_fn)
            res_fn = self.apply_bc(res_fn)
            return res_fn(dofs)

        # Remove unflatten_fn_sol_list
        # Will need to fix
        def constraint_fn_sol_to_sol(sol_list, params):
            """Add documentation

            Args:
                degree (_type_): _description_

            Returns:
                _type_: _description_
            """

            dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
            con_vec = constraint_fn(dofs, params)
            # Not sure what the reshape was for
            return con_vec #.reshape(-1, self.problem.dim[0])

        def get_partial_params_c_fn(sol_list):
            """c(u=u, p)
            """
            def partial_params_c_fn(params):
                return constraint_fn_sol_to_sol(sol_list, params)

            return partial_params_c_fn

        def get_vjp_contraint_fn_params(params, sol_list):
            """v*(partial dc/dp)
            """
            partial_c_fn = get_partial_params_c_fn(sol_list)

            def vjp_linear_fn(v_list):
                primals, f_vjp = jax.vjp(partial_c_fn, params)
                val, = f_vjp(v_list)
                return val

            return vjp_linear_fn

        self.problem.set_params(params)
        res, V = self.problem.newton_update_helper(sol, self.problem.internal_vars, self.problem.internal_vars_surfaces)
        V = self.reduceV(V)

        adjoint_vec, adj_res = self.adjoint_solve(V, self.J, self.I, v,
                                                  np.zeros_like(v))
        # Expose the adjoint linear-solve residual so callers can judge the
        # quality of the reduced gradient: a poorly converged adjoint solve
        # yields a wrong gradient, and unlike a poorly converged forward solve
        # there is no residual downstream that would reveal it. Stored as the
        # raw device scalar; the caller converts it at log time, so no extra
        # synchronization is forced here.
        self._last_adjoint_res = adj_res

        vjp_linear_fn = get_vjp_contraint_fn_params(params, sol)
        # Not sure what the reshape was for
        vjp_result = vjp_linear_fn(adjoint_vec) #.reshape(-1, self.problem.dim[0]))
        vjp_result = jax.tree.map(lambda x: -x, vjp_result)

        return vjp_result

    def ad_wrapper(self, linear=False, use_petsc=False, petsc_options=None, use_petsc_adjoint=False, petsc_options_adjoint=None):
        """Wraps the solver which changes the gradient calculated for 
        functions where solver is called.

        #TODO: May want to allow inputs to tune the inputs of the solve call
        inside the fwd_pred rather than having to use the defaults
        Args:
            linear (bool, optional): #TODO: Remove, not used?
            use_petsc (bool, optional): #TODO: Remove Petsc?
            petsc_options (_type_, optional): #TODO: Remove Petsc?
            use_petsc_adjoint (bool, optional): #TODO: Remove Petsc?
            petsc_options_adjoint (_type_, optional): #TODO: Remove Petsc?

        Returns:
            _type_: _description_
        """
        @jax.custom_vjp
        def fwd_pred(params):
            self.problem.set_params(params)
            # initial_guess = self.problem.initial_guess if hasattr(self.problem, 'initial_guess') else None
            sol_list, info = self.solve()
            assert info[0]
            return sol_list

        def f_fwd(params):
            sol_list = fwd_pred(params)
            return sol_list, (params, sol_list)

        def f_bwd(res, v):
            logger.info("Running backward and solving the adjoint problem...")
            params, sol_list = res
            vjp_result = self.implicit_vjp(sol_list, params, v, use_petsc_adjoint, petsc_options_adjoint)
            return (vjp_result, )

        fwd_pred.defvjp(f_fwd, f_bwd)
        return fwd_pred