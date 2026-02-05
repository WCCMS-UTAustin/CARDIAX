"""

Functions that help with preconditioning.

Need to consider how to organize in the context of using
different solver backends.

"""

import jax.numpy as np

import lineax as lx
import jax
import jax.experimental.sparse as js

import functools as fctls

# datatypes
from jaxtyping import ArrayLike
from cardiax.operators import SparseMatrixLinearOperator

def _set_preconditioning_method(precond: str, A: SparseMatrixLinearOperator):
    """ returns preconditioning method

    Parameters
    ----------
    precond : bool
        turns diagonal preconditioning on and off
    A : js.BCOO
        stiffness matrix outline for the problem
    """

    # check to see if there's a better way of doing this.
    if precond == 'jacobi':
        # jit-compile certain aspects of this function
        mat_diag = np.where(A.matrix.indices[:,0] == A.matrix.indices[:,1], True, False)
        jax_get_diag_fn = jax.jit(fctls.partial(jax_get_diag, mat_diag))
        
        # the partialed version of this function will accept A_sp: js.BCOO as input.
        get_jacobi_precond = fctls.partial(_get_jacobi_precond, jax_get_diag_fn)
        return get_jacobi_precond
    elif precond == 'none':
        # no preconditioning - get identity linear operator.
        I = lx.IdentityLinearOperator(input_structure=A.in_structure(), output_structure=A.out_structure())
        get_no_precond = fctls.partial(_get_no_precond, I)

        # return a partialed version of _get_no_preconditioner
        # that has a fixed identity linear operator.
        return get_no_precond
    else:
        raise NotImplementedError(f"{precond} is not an implemented preconditioner!")

#########################################################
## helpers for the 'no preconditioning' option (False) ##
#########################################################
def _get_no_precond(I: lx.IdentityLinearOperator, A_sp: js.BCOO):

    return I

##############################################################
## helpers for the 'diagonal preconditioning' option (True) ##
##############################################################
def _get_jacobi_precond(jax_get_diag_fn: callable, A_sp: js.BCOO):
    """Creates the action of the jacobi preconditioner
    through the use of the FunctionLinearOperator in lineax.

    Args:
        A_sp (BCOO): jax.experimental.sparse.BCOO object

    Returns:
        jacobian_precond (FunctionLinearOperator): the action of the jacobi preconditioner
    """

    # jax_get_diag_fn is a partialed version of jax_get_diag
    jacobi = jax_get_diag_fn(A_sp)

    jacobi_precond = lx.FunctionLinearOperator(lambda x: x/jacobi, tags=lx.positive_semidefinite_tag, input_structure=jax.eval_shape(lambda: jacobi))
    # def jacobi_precond(x):
    #     return x * (1. / jacobi)

    return jacobi_precond

# some helper functions, might eventually want to move these around
def jax_get_diag(checks: ArrayLike, A_sp: js.BCOO) -> ArrayLike:
    """Function to get the diagonal of a 
    jax.experimental.sparse.BCOO object. This is used
    to compute jacobi preconditioner

    Args:
        checks (np.array(int)): Masked array to bool diagonal indices
        A_sp (BCOO): Sparse matrix object

    Returns:
        diag (np.array): The values along the diagonal of the matrix
    """
    diag = np.ones(A_sp.shape[0])
    diag = diag.at[A_sp.indices[checks,0]].set(A_sp.data[checks], indices_are_sorted=True)
    return diag
