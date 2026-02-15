
import jax.numpy as np
import jax

from cardiax import Problem


class PDE(Problem):
    """This is a MooneyRivlin material model.

    Args:
        Problem (_type_): _description_
    """
    def get_tensor_map(self):

        def psi(F):
            kappa = (4. * (1 + self.nu) * (self.C10 + self.C01)) / (3. * (1. - 2. * self.nu))

            J = np.linalg.det(F)
            C = F.T @ F
            I1 = np.trace(C) 
            Jinv1 = 1/(np.cbrt(J**2))
            Jinv2 = 1/(np.cbrt(J**4))
            I1_bar = I1 * Jinv1
            I2 = 0.5 * ((I1)**2. - np.trace(C@C)) 
            I2_bar = I2 * Jinv2
            energy = self.C10 * (I1_bar - 3.) + self.C01 * (I2_bar - 3.) +  (kappa / 2.) * (J - 1.)**2
         
            return energy

        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad,w3):
            I_d = np.eye(u_grad.shape[0])
            F = u_grad + I_d
            P = P_fn(F)
            return P

        return first_PK_stress
        
    def params(self, params):
        self.C10 = params.get('C10')
        self.C01 = params.get('C01')
        self.nu = params.get('nu')
        return