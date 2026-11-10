"""Forward Dirichlet-elimination regression on an inhomogeneous simple-shear
hyperelasticity problem.

The whole deformation here is imposed through inhomogeneous Dirichlet boundary
conditions, so recovering the analytic deformation gradient exercises the
solver's Dirichlet-elimination and boundary-load handling end to end. A second
check asserts that a single Newton step already satisfies the prescribed
boundary values, which is the signature of the eliminated operator's identity
Dirichlet rows.

The analytic check is run with two preconditioner choices so that the operator
itself is exercised both with and without a preconditioner masking any
elimination error:

* ``precond="none"`` isolates the constrained tangent operator (no preconditioner
  crutch), and
* ``precond="jacobi"`` exercises the usual diagonal-preconditioned path.

This uses only the public solver API and an analytic reference, so it is robust
to changes in how Dirichlet conditions are imposed internally.
"""
import numpy as onp
import numpy.testing as onptest
import jax
import jax.numpy as np
import unittest

from cardiax import box_mesh
from cardiax import FiniteElement, Problem, Newton_Solver
from cardiax import get_F

jax.config.update("jax_enable_x64", True)

GAMMA = 0.1


class HyperElasticity(Problem):
    def get_tensor_map(self):
        def psi(F):
            mu = 50.
            kappa = 5e3
            C1 = mu / 2
            D1 = kappa / 2
            J = np.linalg.det(F)
            I1 = np.trace(F.T @ F)
            return C1 * (I1 - 3. - 2 * np.log(J)) + D1 * (J - 1.)**2.
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            F = u_grad + np.eye(3)
            return P_fn(F)
        return first_PK_stress


def _build_problem():
    """Inhomogeneous simple-shear hyperelasticity on a unit cube."""
    Lx, Ly, Lz = 1., 1., 1.
    mesh = box_mesh(Nx=6, Ny=6, Nz=6, Lx=Lx, Ly=Ly, Lz=Lz)
    fe = FiniteElement(mesh, vec=3, dim=3, ele_type='hexahedron',
                       quadrature_order=1)

    def left(p):   return np.isclose(p[0], 0., atol=1e-5)
    def right(p):  return np.isclose(p[0], Lx, atol=1e-5)
    def front(p):  return np.isclose(p[1], 0., atol=1e-5)
    def back(p):   return np.isclose(p[1], Ly, atol=1e-5)
    def bottom(p): return np.isclose(p[2], 0., atol=1e-5)
    def top(p):    return np.isclose(p[2], Lz, atol=1e-5)

    def zero_bc(p):  return 0.
    def gamma_bc(p): return GAMMA
    def side_bc(p):  return p[1] * GAMMA

    sides_bc = [side_bc, zero_bc, zero_bc]
    info = {"u": [[[left]*3,  [0, 1, 2], sides_bc],
                  [[right]*3, [0, 1, 2], sides_bc],
                  [[front]*3, [0, 1, 2], [zero_bc]*3],
                  [[back]*3,  [0, 1, 2], [gamma_bc, zero_bc, zero_bc]],
                  [[top]*3,   [0, 1, 2], sides_bc],
                  [[bottom]*3,[0, 1, 2], sides_bc]]}
    return HyperElasticity({"u": fe}, dirichlet_bc_info=info)


class Test(unittest.TestCase):
    def test_dirichlet_residual_vanishes_after_one_iteration(self):
        """One Newton step satisfies the Dirichlet conditions exactly.

        Symmetric elimination gives the constrained operator exact identity
        rows at the Dirichlet dofs, with the target increment on the
        right-hand side there. A single undamped Newton step therefore lands
        the solution on the prescribed boundary values, and the Dirichlet block
        of the constrained residual - which holds ``u - u_b`` - drops from the
        full boundary displacement to the linear solver's tolerance.
        """
        problem = _build_problem()
        solver = Newton_Solver(
            problem, np.zeros(problem.num_total_dofs_all_vars),
            line_search_flag=False, precond="none")
        dirichlet_dofs = onp.asarray(problem.bc_inds)

        # Starting from zero, the Dirichlet block of the residual is the whole
        # prescribed displacement, which is what makes the assertion below
        # meaningful rather than a statement about an already-tiny quantity.
        res_before, _ = solver.newton_update_helper(
            np.zeros(problem.num_total_dofs_all_vars),
            problem.internal_vars, problem.internal_vars_surfaces)
        self.assertGreater(
            onp.linalg.norm(onp.asarray(res_before)[dirichlet_dofs]), 1e-2)

        _, info = solver.solve(1e-8, max_iter=1)
        res_after = onp.asarray(info[5])
        onptest.assert_allclose(res_after[dirichlet_dofs],
                                onp.zeros(dirichlet_dofs.size), atol=1e-9)

    def test_forward_recovers_analytic_shear(self):
        """Forward solve reproduces the analytic simple-shear deformation."""
        F_analytic = np.array([[1., GAMMA, 0.],
                               [0., 1., 0.],
                               [0., 0., 1.]])
        for precond in ("none", "jacobi"):
            with self.subTest(precond=precond):
                problem = _build_problem()
                solver = Newton_Solver(
                    problem, np.zeros(problem.num_total_dofs_all_vars),
                    line_search_flag=True, precond=precond)
                sol, info = solver.solve(1e-8, max_iter=40)
                self.assertTrue(info[0],
                                f"forward solve did not converge (precond={precond})")
                Fs = get_F(problem.fes["u"], onp.asarray(sol))
                onptest.assert_allclose(Fs.mean(axis=(0, 1)), F_analytic,
                                        atol=1e-8)


if __name__ == '__main__':
    unittest.main()
