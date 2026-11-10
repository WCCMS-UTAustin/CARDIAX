
import jax.numpy as np

################## assemble a Problem class instance ################
def add_surface_kernels(Problem, surface_kernels):

    class NewProblem(Problem):
        """Add documentation

        Args:
            degree (_type_): _description_

        Returns:
            _type_: _description_
        """
        def __init__(self, *args, **kwargs):
            self.surface_kernels = surface_kernels
            super().__init__(*args, **kwargs)

        def get_surface_maps(self):
            surface_maps_dict = {}
            for fe_var in self.fes:
                surface_maps_temp = {}
                for surf_key in self.surface_kernels[fe_var]:
                    surface_kernel = self.surface_kernels[fe_var][surf_key]
                    if surface_kernel["type"] == "Neumann":
                        if surface_kernel["static"]:
                            surface_maps_temp[surf_key] = create_neumann_kernel_static(surface_kernel["value"])
                        else:
                            surface_maps_temp[surf_key] = create_neumann_kernel_functional()
                    elif surface_kernel["type"] == "Pressure":
                        if surface_kernel["static"]:
                            surface_maps_temp[surf_key] = create_pressure_kernel_static(surface_kernel["value"])
                        else:
                            surface_maps_temp[surf_key] = create_pressure_kernel_functional()
                    elif surface_kernel["type"] == "Spring":
                        if surface_kernel["static"]:
                            surface_maps_temp[surf_key] = create_spring_kernel_static(surface_kernel["value"])
                        else:
                            surface_maps_temp[surf_key] = create_spring_kernel_functional()

                    else:
                        raise ValueError("Surface kernel type not supported. Please choose between Neumann, Pressure, and Spring.")
                surface_maps_dict[fe_var] = surface_maps_temp
            return surface_maps_dict

    return NewProblem

def create_neumann_kernel_static(value):
    """Add documentation

    Args:
        degree (_type_): _description_

    Returns:
        _type_: _description_
    """

    def neumann_kernel(u, u_grad, x):
        return np.array(value)

    return neumann_kernel

def create_neumann_kernel_functional():
    """Add documentation

    Args:
        degree (_type_): _description_

    Returns:
        _type_: _description_
    """

    def neumann_kernel(u, u_grad, x, vec):
        return vec

    return neumann_kernel

def create_pressure_kernel_static(value):
    """Add documentation

    Args:
        degree (_type_): _description_

    Returns:
        _type_: _description_
    """

    def pressure_kernel(u, u_grad, x, normal):
        F = u_grad + np.eye(len(x))
        J = np.linalg.det(F)
        F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
                                [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
                                [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])
        val = value * 133.322/(100**2) * J * F_inv.T @ normal.reshape(3, 1)
        return val.reshape(3)
    
    return pressure_kernel

def create_pressure_kernel_functional():
    """Add documentation

    Args:
        degree (_type_): _description_

    Returns:
        _type_: _description_
    """

    # TODO: make unit conversion part of config
    # Value input as mmHg
    # * 133.322 -> kPa
    # / (100**2) -> N/(cm^2)
    def pressure_kernel(u, u_grad, x, normal, value):
        F = u_grad + np.eye(len(x))
        J = np.linalg.det(F)
        F_inv = 1/J * np.array([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1]],
                                [F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2]],
                                [F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]]).T
        val = value * 133.322 / (100**2) * J * F_inv.T @ normal
        return val
    
    return pressure_kernel

def create_empty_surface_kernel():
    def pressure_kernel(u, u_grad, x, normal, value):
        
        return np.zeros(len(x))
        #return val.reshape(3)
    
    return pressure_kernel

def create_spring_kernel_static(value):
    """Add documentation

    Args:
        degree (_type_): _description_

    Returns:
        _type_: _description_
    """

    def spring_kernel(u, u_grad, x):
        return value * u

    return spring_kernel

def create_spring_kernel_functional():
    """Add documentation

    Args:
        degree (_type_): _description_

    Returns:
        _type_: _description_
    """

    def spring_kernel(u, u_grad, x, value):
        return value * u

    return spring_kernel

def safeguarded(cls):

    """Implements clamped safeguards to prevent Nans from occurring due to a inverting element
    or a large exponential. This should be used with extreme caution as it may change
    the trajectory of the Newton solve but has shown to allow solvers to converge without
    adding extra catches.

    Returns:
        class: Safeguarded class that overwrites the computation of J and exponential evaluation.
    """

    class Safeguarded(cls):
        def set_params(self, params, safeguards=None):
            super().set_params(params)
            sg = safeguards or {}
            self.J_min = sg.get('J_min', 1e-1)
            self.exp_max = sg.get('exp_max', 20.)

        def evaluate_J(self, F):
            return np.maximum(super().evaluate_J(F), self.J_min)

        def evaluate_exp(self, x):
            return super().evaluate_exp(np.minimum(x, self.exp_max))

    Safeguarded.__name__ = f"_{cls.__name__}"
    return Safeguarded