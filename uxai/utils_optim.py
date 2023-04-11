""" Implementation of optimization routines """

import numpy as np
from scipy.optimize import bisect
from scipy.linalg import eigh
import trustregion
from warnings import warn



def opt_lin_ellipsoid_quadrant(a, ellipsoid_dict, monotocity, epsilon, constraint=True):
    
    minmax_preds, arg_minmax = ellipsoid_dict[()].opt_linear(a, epsilon, return_input_sol=True)
    # Make sure that the optimum respect the monotocity contraints
    if constraint and (arg_minmax * monotocity.reshape((1, 1, -1)) > 0).any():
        # Min and Max
        for i in [0, 1]:
            mismatch = arg_minmax[i] * monotocity > 0
            broken_consts, inverse_idx = np.unique(mismatch, axis=0, return_inverse=True)
            # For each set of broken constraints we resolve the problem with
            # a slice of the original ellipsoid
            for j, broken_const in enumerate(broken_consts):
                # Skip this trivial case
                if np.sum(broken_const) == 0:
                    continue
                # Inputs that lead to an invalid constraint
                where_invalid = np.where(inverse_idx == j)[0]
                # Ellipsoid has already been sliced that way
                if not tuple(broken_const) in ellipsoid_dict:
                    ellipsoid_dict[tuple(broken_const)] = ellipsoid_dict[()].slice(broken_consts)
                sliced_ellipsoid = ellipsoid_dict[tuple(broken_const)]
                # Sliced obj
                new_a = a[respect_const, where_invalid]
                minmax_preds[where_invalid, i] = sliced_ellipsoid.opt_linear(new_a, epsilon)[:, i]
    return minmax_preds



def qpqc(W, alpha_hat):
    """ Routine for qpqc """

    if np.min(W) <= 0:
        if np.isclose(alpha_hat[np.argmin(W)], 0).all():
            raise Exception("Hard Case !!! Don't use the exact solver.")

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
        # We start the bisection search pasted the absolute value 
        # |w_i| for the most negative eigenvalue.
        else:
            w_max = np.max(np.abs(W))
            w_discont = -np.min(W)
            alpha_discont = alpha_hat[np.argmin(W)]
            start = w_discont * (alpha_discont/ np.sqrt(2) + 1)
            assert f(start) > 0, "Wrong starting point"
            end = (norm_alpha_hat + 1) * w_max
            while f(end) >= 0:
                end *= 2
        
        # Bisect search for lambda
        lambd_star, res = bisect(f, start, end, maxiter=500, full_output=True)
        # Make sure bisection has converged
        assert res.converged
        argmin = (W / (W + lambd_star)).reshape((-1, 1)) * alpha_hat
        # Make sure we are on the border when necessary
        assert np.isclose(np.linalg.norm(argmin), 1)
    
    return argmin



def qpqc_2(W, b_hat):
    """ Routine for qpqc """

    if np.min(W) <= 0:
        if np.isclose(b_hat[W==np.min(W)], 0).all():
            raise Exception("Hard Case !!! Don't use the exact solver.")

    b_hat = -0.5 * b_hat
    norm_b_hat = np.linalg.norm(b_hat)

    # Function representing the norm of b
    def f(lambd):
        return np.linalg.norm(1 / (W + lambd) * b_hat.ravel()) ** 2 - 1

    # No discontinuities in f       
    if (W>0).all():
        start = 0
        end = norm_b_hat - np.min(W)
        # Trivial solution
        if f(start) < 0:
            argmin = (1 / W).reshape((-1, 1)) * b_hat
            return argmin
    # Discontunities associated with dims with w_i < 0
    # We start the bisection search pasted the absolute value 
    # |w_i| for the most negative eigenvalue.
    else:
        w_max = np.max(np.abs(W))
        w_discont = -np.min(W)
        b_discont = b_hat[np.argmin(W)]
        start = b_discont / np.sqrt(2) + w_discont
        assert f(start) > 0, "Wrong starting point"
        end = norm_b_hat + w_max
        while f(end) >= 0:
            end *= 2
    
    # Bisect search for lambda
    lambd_star, res = bisect(f, start, end, maxiter=500, full_output=True)
    # Make sure bisection has converged
    assert res.converged
    argmin = (1 / (W + lambd_star)).reshape((-1, 1)) * b_hat
    # Make sure we are on the border when necessary
    assert np.isclose(np.linalg.norm(argmin), 1)

    return argmin



def opt_qpqc_centered_exact(A, x_hat):
    """ 
    Compute the min/argmin and max/argmax of 
    g(x) = (x - x_hat)^T A (x - x_hat) with the
    constraint that x^T x <= 1

    Parameters
        A: (d, d) symmetric matrix
        x_hat: (d, 1) array

    Returns
        min_val: float
        argmin: (d, 1) array
        max_val: float
        argmax: (d, 1) array
    """
    # Eigen decomposition of quadratic form
    W, V = eigh(A)
    
    # Change of basis for x_hat
    alpha_hat = V.T.dot(x_hat)
    sol = np.zeros_like(alpha_hat)

    # Some eigenvalues can be null
    zero_idx = list(np.where(W == 0)[0])
    non_zero_idx = [i for i in range(len(W)) if i not in zero_idx]
    if len(non_zero_idx) == 0:
        return 0, sol, 0, sol
    
    # Minimize
    sol[non_zero_idx] = qpqc(W[non_zero_idx], alpha_hat[non_zero_idx])
    argmin = V.dot(sol)
    min_val = float( (argmin - x_hat).T.dot(A.dot(argmin - x_hat)) )

    # Maximize
    sol[non_zero_idx] = qpqc(-W[non_zero_idx], alpha_hat[non_zero_idx])
    argmax = V.dot(sol)
    max_val = float( (argmax - x_hat).T.dot(A.dot(argmax - x_hat)) )

    return min_val, argmin, max_val, argmax



def opt_qpqc_standard_exact(A, b):
    """ 
    Compute the min/argmin and max/argmax of 
    `g(x) = x^T A x + b^T x`
    subject to `x^Tx <= 1`

    Parameters
    ----------
    A: (d, d) symmetric matrix
    b: (d, 1) vector
    x_hat: (d, 1) array

    Returns
    -------
    min_val: float
    argmin: (d, 1) array
    max_val: float
    argmax: (d, 1) array
    """
    
    # Eigen decomposition of quadratic form
    W, V = eigh(A)
    
    # Change of basis
    b_hat = V.T.dot(b)

    # Minimize
    argmin = V.dot(qpqc_2(W, b_hat))
    min_val = float( argmin.T.dot(A.dot(argmin)) + b.T.dot(argmin) )

    # Maximize
    argmax = V.dot(qpqc_2(-W, -b_hat))
    max_val = float( argmax.T.dot(A.dot(argmax)) + b.T.dot(argmax) )

    return min_val, argmin, max_val, argmax



def opt_qpqc_standard_approx(A, b):
    """ 
    Compute the min/argmin and max/argmax of 
    `g(x) = x^T A x + b^T x`
    subject to `x^T x <= 1`

    Parameters
    ----------
    A: (d, d) symmetric matrix
    b: (d, 1) vector

    Returns
    -------
    min_val: float
    max_val: float
    """
    
    # Run the QPQC in "trust region subproblem" form
    argmin = trustregion.solve(b.ravel(), 2*A, 1)
    min_val = 0.5 * argmin.T.dot(A.dot(argmin)) + b.T.dot(argmin)
    argmax = trustregion.solve(-b.ravel(), -2*A, 1)
    max_val = 0.5 * argmax.T.dot(A.dot(argmax)) + b.T.dot(argmax)

    return float(min_val), float(max_val)