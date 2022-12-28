""" Assert that QPQC returns right shapes and values """

import pytest
import numpy as np
from scipy.stats import chi2

import sys, os
sys.path.append(os.path.join('../..'))
from uxai.utils_optim import opt_qpqc_standard_exact, opt_qpqc


@pytest.mark.parametrize("d", range(2, 10))
def test_qpqc_shape(d):
    
    W = np.random.uniform(-d, d, size=(d,))
    A = np.diag(W)
    x_hat = np.random.uniform(-1, 1, size=(d, 1))

    # Standard form
    min_val, argmin, max_val, argmax = opt_qpqc(A, x_hat)
    assert type(min_val) == float, "Standard QPQC does not return the right type"
    assert type(max_val) == float, "Standard QPQC does not return the right type"
    assert argmin.shape == (d, 1), "Standard QPQC does not return the right type"
    assert argmax.shape == (d, 1), "Standard QPQC does not return the right type"

    # Non-Standard Form
    C = np.diag(np.random.uniform(1, d, size=(d,)))
    b = np.random.uniform(1, 2, size=(d, 1))
    min_val, argmin, max_val, argmax = opt_qpqc_standard_exact(A, b, C, x_hat)
    assert type(min_val) == float, "QPQC does not return the right type"
    assert type(max_val) == float, "QPQC does not return the right type"
    assert argmin.shape == (d, 1), "QPQC does not return the right type"
    assert argmax.shape == (d, 1), "QPQC does not return the right type"



@pytest.mark.parametrize("d", range(2, 100, 5))
def test_qpqc_values_non_standard(d):
    # Standard
    for _ in range(100):
        W = np.random.uniform(-d, d, size=(d,))
        A = np.diag(W)
        x_hat = np.random.uniform(-1, 1, size=(d, 1))

        def f(X):
            X_tilde = X - x_hat.T
            return np.sum(X_tilde.dot(A) * X_tilde, axis=1, keepdims=True)

        min_val, _, max_val, _ = opt_qpqc(A, x_hat)

        # Sample points from the unit sphere
        z = np.random.normal(0, 1, size=(10000, d)) / np.sqrt(chi2.ppf(0.5, df=d))
        z = z[np.linalg.norm(z, axis=1) <= 1]

        # Evaluate the model on all these points
        fz = f(z)

        assert min_val <= np.min(fz), "The minimum is not the smallest value"
        assert max_val >= np.max(fz), "The maxmum is not the largest value"



@pytest.mark.parametrize("d", range(2, 100, 5))
def test_qpqc_values_standard(d):
    # Non-Standard
    for _ in range(100):
        W = np.random.uniform(-d, d, size=(d,))
        A = np.diag(W)
        x_hat = np.random.uniform(-1, 1, size=(d, 1))
        semi_lengths = np.sqrt(np.random.uniform(1, d, size=(d,)))
        C = np.diag(semi_lengths**2)
        b = np.random.uniform(1, 2, size=(d, 1))

        # x^T A x + b^T x
        def f(X):
            out = np.sum(X.dot(A) * X, axis=1, keepdims=True)
            out += X.dot(b)
            return out

        min_val, _, max_val, _ = opt_qpqc_standard_exact(A, b, C, x_hat)

        # Sample points from the unit sphere
        z = np.random.normal(0, 1, size=(10000, d)) / np.sqrt(chi2.ppf(0.5, df=d))
        z = z[np.linalg.norm(z, axis=1) <= 1]
        z = z / semi_lengths + x_hat.T

        # Evaluate the model on all these points
        fz = f(z)
        print("Predicted : ", min_val, max_val)
        print("Actual : ", np.min(fz), np.max(fz))
        assert min_val <= np.min(fz), "The minimum is not the smallest value"
        assert max_val >= np.max(fz), "The maxmum is not the largest value"



if __name__ == "__main__":
    test_qpqc_values_standard(10)