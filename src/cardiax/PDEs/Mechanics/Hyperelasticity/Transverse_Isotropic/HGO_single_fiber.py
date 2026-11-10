
import jax.numpy as np
import jax
from cardiax import Problem

class PDE(Problem):
    """HGO single fiber family hyperelastic material model (Holzapfel-Gasser-Ogden 2000).

    Strain energy: psi = psi_iso + psi_fiber + psi_vol
      psi_iso   = (c/2) * (I1_bar - 3)
      psi_fiber = (k1/2k2) * [exp(k2*(I4_bar - 1)^2) - 1]  (active only in tension, I4_bar > 1)
      psi_vol   = (kappa/2) * (J - 1)^2

    All invariants are computed from the isochoric right Cauchy-Green tensor C_bar = J^(-2/3) * C.
    """
    def get_tensor_map(self):

        def psi(F, a0):
            a0 = a0 / np.linalg.norm(a0)
            J = self.evaluate_J(F)
            C = F.T @ F
            C_bar = J**(-2./3.) * C

            I1_bar = np.trace(C_bar)
            I4_bar = a0 @ C_bar @ a0

            psi_iso = (self.c / 2.) * (I1_bar - 3.)
            psi_vol = (self.kappa / 2.) * (J - 1.)**2.
            fiber_term = (self.k1 / (2. * self.k2)) * (self.evaluate_exp(self.k2 * (I4_bar - 1.)**2.) - 1.)
            psi_fiber = np.where(I4_bar > 1., fiber_term, 0.)

            return psi_iso + psi_fiber + psi_vol

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, a0):
            I_d = np.eye(u_grad.shape[0])
            F = u_grad + I_d
            P = P_fn(F, a0)
            return P

        return first_PK_stress

    def set_params(self, params):
        self.c = params.get('c', 1.)       # ground matrix parameter (MPa)
        self.k1 = params.get('k1', 1.)     # fiber stiffness (MPa)
        self.k2 = params.get('k2', 1.)     # fiber nonlinearity (-)
        self.kappa = params.get('kappa', 10.) # bulk modulus (MPa)
        return
