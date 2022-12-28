""" Implementation of optimization routines """

import numpy as np
from scipy.optimize import bisect
from scipy.linalg import eigh
import trustregion



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



def qpqc(W, alpha_hat):
    """ Routine for qpqc """
    if (alpha_hat == 0).any():
        raise ValueError("The QPQC is degenerate")
    
    norm_alpha_hat = np.linalg.norm(alpha_hat)

    # Scenario 1 (Trivial)
    if norm_alpha_hat <= 1 and (W>0).all():
        # Min is trivial
        argmin = alpha_hat
    # Scenario 2 (Non-Trivial)
    else:
        # Function representing the norm of alpha
        def f(lambd):
            return np.linalg.norm(W / (W + lambd) * alpha_hat.ravel()) ** 2 - 1

        # No discontinuities in f       
        if (W>0).all():
            start = 0
            end = (norm_alpha_hat - 1) * np.max(W)
        # Discontunities associated with dims with w_i < 0
        else:
            w_max = np.max(np.abs(W))
            w_start = -np.min(W)
            alpha_start = alpha_hat[np.argmin(W)]
            start = w_start * (alpha_start / np.sqrt(2) + 1)
            assert f(start) > 1
            end = (norm_alpha_hat + 1) * w_max
        
        # Bisect search for lambda
        lambd_star, res = bisect(f, start, end, maxiter=500, full_output=True)
        # Make sure bissection has converged
        assert res.converged
        argmin = (W / (W + lambd_star)).reshape((-1, 1)) * alpha_hat
        # Make sure we are on the border when necessary
        assert np.isclose(np.linalg.norm(argmin), 1)
    
    return argmin


def opt_qpqc(A, x_hat):
    """ 
    Compute the min/argmin and max/argmax of 
    g(x) = (x - x_hat)^T A (x - x_hat) with the
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
    W, V = eigh(A)
    assert np.abs(W).min() > 1e-11, "Matrix must be non-singular"
    
    # Change of basis for x_hat
    alpha_hat = V.T.dot(x_hat)
    
    # Minimize
    argmin = V.dot(qpqc(W, alpha_hat))
    min_val = float( (argmin - x_hat).T.dot(A.dot(argmin - x_hat)) )

    # Maximize
    argmax = V.dot(qpqc(-W, alpha_hat))
    max_val = float( (argmax - x_hat).T.dot(A.dot(argmax - x_hat)) )

    return min_val, argmin, max_val, argmax



def opt_qpqc_standard_exact(A, b, C, x_hat):
    """ 
    Compute the min/argmin and max/argmax of 
    `g(x) = x^T A x + b^T x`
    subject to `(x-x_hat)^T C (x-h_hat) <= 1`

    Parameters
    ----------
    A: (d, d) non-singular symmetric matrix
    b: (d, 1) vector
    C: (d, d) positive definite matrix
    x_hat: (d, 1) array

    Returns
    -------
    min_val: float
    argmin: (d, 1) array
    max_val: float
    argmax: (d, 1) array
    """
    
    C_half = np.linalg.cholesky(C)
    C_half_inv = np.linalg.inv(C_half)
    # Complete the square of the objective
    if (b!=0).any():
        x_prime = -0.5 * np.linalg.solve(A, b)
        gap = -x_prime.T.dot(A.dot(x_prime))
        z_hat = C_half.T.dot(x_prime - x_hat)
    else:
        gap = 0
        z_hat = -1 * C_half.T.dot(x_hat)
    A_prime = C_half_inv.dot(A.dot(C_half_inv.T))
    
    # Run the QPQC in "trust region subproblem" form
    min_val, argmin, max_val, argmax = opt_qpqc(A_prime, z_hat)
    min_val += gap
    max_val += gap
    # Inverse transform
    argmin = C_half_inv.T.dot(argmin) + x_hat
    argmax = C_half_inv.T.dot(argmax) + x_hat

    return float(min_val), argmin, float(max_val), argmax


def opt_qpqc_standard_approx(A, b, C, x_hat):
    """ 
    Compute the min/argmin and max/argmax of 
    `g(x) = x^T A x + b^T x`
    subject to `(x-x_hat)^T C (x-h_hat) <= 1`

    Parameters
    ----------
    A: (d, d) symmetric matrix, can be singular
    b: (d, 1) vector
    C: (d, d) positive definite matrix
    x_hat: (d, 1) array

    Returns
    -------
    min_val: float
    argmin: (d, 1) array
    max_val: float
    argmax: (d, 1) array
    """

    C_half = np.linalg.cholesky(C)
    C_half_inv = np.linalg.inv(C_half)
    # Complete the square of the objective
    A_prime = C_half_inv.dot(A.dot(C_half_inv.T))
    b_prime = C_half_inv.dot(b + 2 * A.dot(x_hat))
    gap = float(x_hat.T.dot(A.dot(x_hat)) + b.T.dot(x_hat))
    # Run the QPQC in "trust region subproblem" form
    argmin = trustregion.solve(b_prime.ravel(), 2*A_prime, 1)
    min_val = 0.5 * argmin.T.dot(A_prime.dot(argmin)) + b_prime.T.dot(argmin)
    argmax = trustregion.solve(-b_prime.ravel(), -2*A_prime, 1)
    max_val = 0.5 * argmax.T.dot(A_prime.dot(argmax)) + b_prime.T.dot(argmax)
    # Add the gap
    min_val += gap
    max_val += gap

    return float(min_val), float(max_val)