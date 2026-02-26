
import jax
import jax.numpy as np

from cardiax import Problem

class PDE(Problem):

    def get_tensor_map(self):
        def psi(F, f, s, n):
            f = f[:, None]
            s = s[:, None]
            n = n[:, None]
            J = np.linalg.det(F)
            C = F.T @ F * np.cbrt(J)**(-2)
            E_tilde = 1/2 * (C - np.eye(self.dim["u"]))
            E11 = f.T @ E_tilde @ f
            E12 = f.T @ E_tilde @ s
            E13 = f.T @ E_tilde @ n
            E22 = s.T @ E_tilde @ s
            E23 = s.T @ E_tilde @ n
            E33 = n.T @ E_tilde @ n

            Q = self.A1 * E11**2 \
            + self.A2 * (E22**2 + E33**2 + 2*E23**2) \
            + self.A3 * (E12**2 + E13**2)
            
            psi_dev = self.c/2 * (np.exp(self.alpha * Q[0, 0]) - 1)
            psi_vol = self.K/2 * ( (J**2 - 1)/2 - np.log(J))
            return psi_dev + psi_vol
        
        P_fn = jax.grad(psi)

        def S_act(F, f, TCa):
            f = f[:, None]
            lamb = np.sqrt(f.T @ F.T @ F @ f)
            S = TCa * (1 + self.beta * (lamb - 1))/(lamb ** 2) * f @ f.T
            return S

        def first_PK_stress(u_grad, f, s, n, TCa):
            F = u_grad + np.eye(self.dim["u"])
            P_psi = P_fn(F, f, s, n)
            P_act = F @ S_act(F, f, TCa)
            return P_psi + P_act

        return first_PK_stress

    def set_params(self, params: dict = {}):
        # Default parameters
        self.c = params.get('c', 1522.083)
        self.A1 = params.get('A1', 12.)
        self.A2 = params.get('A2', 8.)
        self.A3 = params.get('A3', 26.)
        self.K = params.get('K', 1e5)
        self.alpha = params.get('alpha', 2.125)
        self.beta = params.get('beta', 1.4)
        return