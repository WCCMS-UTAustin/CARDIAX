
import jax.numpy as np
import jax
from cardiax import Problem

class PDE(Problem):
    """This is a Yeoh material model.

    Args:
        Problem (_type_): _description_
    """
    def get_tensor_map(self):

        def psi(F):
            J = self.evaluate_J(F)
            Jinv = 1/(np.cbrt(J**2))
            I1 = np.trace(F.T @ F)
            e1 = I1 - 3.
            energy = self.c1 * e1 + self.c2 * e1**2 + self.c3 * e1**3
            energy += self.K/2 * (J - 1.)**2
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I_d = np.eye(u_grad.shape[0])
            F = u_grad + I_d
            P = P_fn(F)
            return P

        return first_PK_stress

    def set_params(self, params):
        self.c1 = params.get('c1', 10.)
        self.c2 = params.get('c2', 0.)
        self.c3 = params.get('c3', 0.)
        self.K = params.get('K', 1e4)
        return