"""

Time integrators for structural dynamics problems.

List of time integration methods that are included in this
file:
    - generalized-alpha (functional)

"""

import numpy as np

######################################################
## functions for generalized alpha time integration ##
######################################################

def get_generalized_alpha_params(alpha_f=0.4, alpha_m=0.2):
    """ computes parameters for generalized_alpha_method

    returns gamma and beta, two constants that are dependent on
    the alpha-type parameters alpha_f and alpha_m.

    this method can be used to partial the functions below
    to avoid passing around alpha_m and alpha_f. I might automate
    this function partialing process...

    Parameters
    ----------
    alpha_f : _type_
        _description_
    alpha_m : _type_
        _description_

    Returns
    -------
    gamma, beta

    """

    gamma = 0.5 - alpha_m + alpha_f
    beta = 0.25 * np.pow((1 - alpha_m + alpha_f), 2)

    return gamma, beta


def get_f_alpha(f, f_old, alpha):
    """ generic function for alpha-level quantities

    computes alpha-level quantity f, which is a convex
    combination of f and f_old.

    Parameters
    ----------
    f : _type_
        _description_
    f_old : _type_
        _description_
    alpha : _type_
        _description_
    """

    # this is the version that's in the paper!        
    return (1.0 - alpha) * f + alpha * f_old

# ALL DEFAULT VALUES ARE WITH ALHPA_F = 0.4 AND ALPHA_M = 0.2
def get_u_dot(u, u_old, u_dot_old, u_ddot_old, delta_t, gamma=0.7, beta=0.36):
    """ computes (predicts) u_dot_n+1

    """

    return (gamma / (beta * delta_t)) * u + (-gamma / (beta * delta_t)) * u_old + (1.0 - gamma / beta) * u_dot_old + ((1.0 - gamma)*delta_t - ((1.0 - 2.0*beta) * delta_t * gamma/(2.0*beta))) * u_ddot_old

def get_u_ddot(u, u_old, u_dot_old, u_ddot_old, delta_t, gamma=0.7):
    """ computes (predicts) u_ddot_n+1

    """

    # get old velocity
    u_dot = get_u_dot(u, u_old, u_dot_old, u_ddot_old, delta_t)
    return (1.0 / (delta_t * gamma)) * u_dot + (-1.0 / (delta_t * gamma)) * u_dot_old + (-(1.0 - gamma) / gamma) * u_ddot_old

def get_u_alpha(u, u_old, alpha_f=0.4):
    return get_f_alpha(u, u_old, alpha_f)

# NOTE: want to avoid if statements, and will only be implementing
#       a second order time integrator for now...
def get_u_dot_alpha(u, u_old, u_dot_old, u_ddot_old, delta_t, alpha_f=0.4):
    
    u_dot = get_u_dot(u, u_old, u_dot_old, u_ddot_old, delta_t)

    return get_f_alpha(u_dot, u_dot_old, alpha_f)

# acceleration!
def get_u_ddot_alpha(u, u_old, u_dot_old, u_ddot_old, delta_t, alpha_m=0.2):

    # again, maybe try to avoid recomputation...
    u_ddot = get_u_ddot(u, u_old, u_dot_old, u_ddot_old, delta_t)

    # alpha_m is used for the acceleration!
    return get_f_alpha(u_ddot, u_ddot_old, alpha_m)

# advance to the next time step (intermediate time step, technically)
def update_time(u, u_old, u_dot_old, u_ddot_old, t, delta_t):
    
    # # update time!
    t_new = t + delta_t

    # return all old quantities - check that this is updating appropriately.
    u_dot = get_u_dot(u, u_old, u_dot_old, u_ddot_old, delta_t)
    u_ddot = get_u_ddot(u, u_old, u_dot_old, u_ddot_old, delta_t)

    # these are n+1 timestep quantities, which are used as old data
    # at the next step!
    return t_new, u, u_dot, u_ddot

def get_initial_params(initial_guess, delta_t):
    """ gets/sets initial conditions.

    given an initial guess, appropriate IC are prescribed - these
    will just be arrays of 0s with the same shape as the initial guess.

    Parameters
    ----------
    initial_guess : _type_
        _description_
    """

    t = delta_t
    u_dot = np.zeros_like(initial_guess)
    u_ddot = np.zeros_like(initial_guess)

    return t, initial_guess, u_dot, u_ddot