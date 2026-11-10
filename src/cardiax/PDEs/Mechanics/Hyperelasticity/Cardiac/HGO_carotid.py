import jax
import jax.numpy as np

from cardiax import Problem

class PDE(Problem):

    def get_tensor_map(self):
        def psi(F, f):
            # Single circumferential fiber family
            I = np.eye(self.dim["u"])
            J = self.evaluate_J(F)
            C = F.T @ F

            # Isochoric neo-Hookean ground matrix
            I1_bar = np.cbrt(J)**(-2) * np.trace(C)
            psi_iso = self.mu / 2 * (I1_bar - np.trace(I))

            # Volumetric penalty: K/4 (J^2 - 1 - 2 ln J)
            psi_vol = self.K / 4 * (J**2 - 1 - 2 * np.log(J))

            # Fiber term on full (not isochoric) C; tension-only switch
            I4 = f @ C @ f
            E4 = np.maximum(I4 - 1., 0.)
            psi_fib = self.k1 / (2 * self.k2) * (self.evaluate_exp(self.k2 * E4**2) - 1)

            return psi_iso + psi_vol + psi_fib

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, f):
            F = u_grad + np.eye(self.dim["u"])
            return P_fn(F, f)

        return first_PK_stress

    def set_params(self, params):
        # Rabbit carotid media, Holzapfel, Gasser & Ogden, J. Elasticity 2000 [kPa]
        # psi_iso = c/2 (I1_bar - 3)  =>  mu = c
        self.mu = params.get('mu', 30.0)
        self.k1 = params.get('k1', 20.0)
        self.k2 = params.get('k2', 5.0)          # dimensionless
        # K = 100*mu (nu ≈ 0.495): near-incompressible without severe locking on linear hexes
        self.K = params.get('K', 300.)
        return