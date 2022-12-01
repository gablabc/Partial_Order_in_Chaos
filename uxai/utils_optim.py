""" Implementation of optimization routines """

import numpy as np
from scipy.optimize import bisect



def opt_lin_ellipsoid(a, A_half_inv, x_hat, return_input_sol=False):
    """ 
    Compute the min/argmin and max/argmax of g(x) = a^t x  with the
    constraint that (x - x_hat)^T A (x - x_hat) <= 1

    Parameters
        a: (d, N) array
        A_half_inv: (d, d) array
        x_hat: (d, 1) array

    Returns
        sol_values: (N, 2)
        sol_inputs: (2, N, d)
    """
    a_prime = A_half_inv.dot(a).T # (N, d)
    norm_a_prime = np.linalg.norm(a_prime, axis=1, keepdims=True) # (N, 1)
    sol_values = np.array([-1, 1]) * norm_a_prime + a.T.dot(x_hat) # (N, 2)
    if not return_input_sol:
        return sol_values
    else:
        z_star = a_prime.dot(A_half_inv) / norm_a_prime # (N, d)
        sol_inputs = np.array([-1, 1]).reshape((2, 1, 1)) * z_star + x_hat.T # (2, N, d)
        return sol_values, sol_inputs



def qpqc(W, x_hat):
    """ Routine for qpqc """
    if (x_hat == 0).any():
        raise ValueError("The QPQC is degenerate")
    
    norm_x_hat = np.linalg.norm(x_hat)
    bound = np.sqrt(np.ceil(norm_x_hat) ** 2)

    # Scenario 1 (Trivial)
    if norm_x_hat <= 1 and (W>0).all():
        # Min is trivial
        argmin = x_hat
    # Scenario 2 (Non-Trivial)
    else:
        def f(lambd):
            return np.linalg.norm(W / (W + lambd) * x_hat.ravel()) ** 2 - 1

        # No discontinuities in f       
        if (W>0).all():
            start = 0
            end = (bound - 1) * np.max(W)
        # Discontunities associated with dims with w_i < 0
        else:
            w_max = np.max(np.abs(W))
            # Go at the highest discontunity with alpha_hat_i not null
            idx = np.where(~np.isclose(x_hat, 0))[0]
            w_start = np.min(W[idx])
            alpha_start = x_hat[idx[np.argmin(W[idx])], 0]
            start = -w_start * (alpha_start / np.sqrt(2) + 1)
            assert f(start) > 1
            end = (bound + 1) * w_max
        
        # Bisect search for lambda
        lambd_star = bisect(f, start, end, maxiter=200)
        argmin = (W / (W + lambd_star)).reshape((-1, 1)) * x_hat

    return argmin


def opt_qpqc(A, x_hat):
    """ 
    Compute the min/argmin and max/argmax of g(x) = (x - x_hat)^T A (x - x_hat) with the
    constraint that x^T x <= 1

    Parameters
        A: (d, d) non-singular matrix
        x_hat: (d, 1) array

    Returns
        min_val: float
        argmin: (d, 1) array
        max_val: float
        argmax: (d, 1) array
    """
    # Eigen decomposition of quadratic form
    W, V = np.linalg.eig(A)
    assert not (W ==0).any(), "Matrix must be non-singular"
    
    # Change of basis for x_hat
    alpha_hat = V.T.dot(x_hat)
    
    # Minimize
    argmin = V.dot(qpqc(W, alpha_hat))
    min_val = float( (argmin - x_hat).T.dot(A.dot(argmin - x_hat)) )

    # Maximize
    argmax = V.dot(qpqc(-W, alpha_hat))
    max_val = float( (argmax - x_hat).T.dot(A.dot(argmax - x_hat)) )

    return min_val, argmin, max_val, argmax