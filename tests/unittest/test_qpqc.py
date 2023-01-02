""" Assert that QPQC returns right shapes and values """

import pytest
import numpy as np
from scipy.stats import chi2, loguniform

import sys, os
sys.path.append(os.path.join('../..'))
from uxai.utils_optim import opt_qpqc_standard_exact, opt_qpqc_centered_exact


@pytest.mark.parametrize("d", range(2, 10))
def test_qpqc_shape(d):
    np.random.seed(42)
    W = np.random.uniform(-d, d, size=(d,))
    A = np.diag(W)
    x_hat = np.random.uniform(-1, 1, size=(d, 1))

    # Standard form
    min_val, argmin, max_val, argmax = opt_qpqc_centered_exact(A, x_hat)
    assert type(min_val) == float, "Standard QPQC does not return the right type"
    assert type(max_val) == float, "Standard QPQC does not return the right type"
    assert argmin.shape == (d, 1), "Standard QPQC does not return the right type"
    assert argmax.shape == (d, 1), "Standard QPQC does not return the right type"

    # Non-Standard Form
    b = np.random.uniform(1, 2, size=(d, 1))
    min_val, argmin, max_val, argmax = opt_qpqc_standard_exact(A, x_hat)
    assert type(min_val) == float, "QPQC does not return the right type"
    assert type(max_val) == float, "QPQC does not return the right type"
    assert argmin.shape == (d, 1), "QPQC does not return the right type"
    assert argmax.shape == (d, 1), "QPQC does not return the right type"



def centered_problem(W):
    d = len(W)
    A = np.diag(W)
    x_hat = np.random.uniform(-1, 1, size=(d, 1))

    # (x - x_hat)^T A (x - x_hat)
    def f(X):
        X_tilde = X - x_hat.T
        return np.sum(X_tilde.dot(A) * X_tilde, axis=1, keepdims=True)

    min_val, _, max_val, _ = opt_qpqc_centered_exact(A, x_hat)

    # Sample points from the unit sphere
    z = np.random.normal(0, 1, size=(50000, d)) / np.sqrt(chi2.ppf(0.5, df=d))
    z = z[np.linalg.norm(z, axis=1) <= 1]

    # Evaluate the model on all these points
    fz = f(z)

    assert min_val <= np.min(fz), "The minimum is not the smallest value"
    assert max_val >= np.max(fz), "The maxmum is not the largest value"


def standard_problem(W):
    d = len(W)
    A = np.diag(W)
    b = np.random.uniform(-1, 1, size=(d, 1))

    # x^T A x + b^T x
    def f(X):
        out = np.sum(X.dot(A) * X, axis=1, keepdims=True)
        out += X.dot(b)
        return out

    # Solve
    min_val, _, max_val, _ = opt_qpqc_standard_exact(A, b)

    # Sample points from the unit sphere
    z = np.random.normal(0, 1, size=(50000, d)) / np.sqrt(chi2.ppf(0.5, df=d))
    z = z[np.linalg.norm(z, axis=1) <= 1]

    # Evaluate the model on all these points
    fz = f(z)
    print("Predicted : ", min_val, max_val)
    print("Actual : ", np.min(fz), np.max(fz), "\n")
    assert min_val <= np.min(fz), "The minimum is not the smallest value"
    assert max_val >= np.max(fz), "The maxmum is not the largest value"



@pytest.mark.parametrize("d", range(50, 1000, 50))
@pytest.mark.parametrize("s", range(10))
def test_qpqc_values_centered_well_cond(d, s):
    # Non-Standard
    np.random.seed(s)
    W = np.random.randint(-d, d, size=(d,))
    centered_problem(W)


@pytest.mark.parametrize("d", range(50, 1000, 50))
@pytest.mark.parametrize("s", range(10))
def test_qpqc_values_centered_well_cond_2(d, s):
    # Non-Standard
    np.random.seed(s)
    W = np.random.uniform(-2*d, 2*d, size=(d,))
    centered_problem(W)


@pytest.mark.parametrize("d", range(50, 1000, 50))
@pytest.mark.parametrize("s", range(10))
def test_qpqc_values_centered_ill_cond(d, s):
    # Non-Standard
    np.random.seed(s)
    W = loguniform.rvs(a=1e-10, b=1e10, size=(d,))
    centered_problem(W)


@pytest.mark.parametrize("d", range(50, 1000, 50))
@pytest.mark.parametrize("s", range(10))
def test_qpqc_values_centered_ill_cond_2(d, s):
    # Non-Standard
    np.random.seed(s)
    W = loguniform.rvs(a=1e-10, b=1e10, size=(d,))
    idx = np.random.choice(d, d//2)
    W[idx] = -1* W[idx]
    centered_problem(W)


@pytest.mark.parametrize("d", range(50, 1000, 50))
@pytest.mark.parametrize("s", range(10))
def test_qpqc_values_standard_well_cond(d, s):
    # Standard
    np.random.seed(s)
    W = np.random.randint(-d, d, size=(d,))
    standard_problem(W)


@pytest.mark.parametrize("d", range(50, 1000, 50))
@pytest.mark.parametrize("s", range(10))
def test_qpqc_values_standard_well_cond_2(d, s):
    # Standard
    np.random.seed(s)
    W = np.random.uniform(-2*d, 2*d, size=(d,))
    standard_problem(W)


@pytest.mark.parametrize("d", range(50, 1000, 50))
@pytest.mark.parametrize("s", range(10))
def test_qpqc_values_standard_ill_cond(d, s):
    # Standard
    np.random.seed(s)
    W = loguniform.rvs(a=1e-6, b=1e6, size=(d,))
    standard_problem(W)


@pytest.mark.parametrize("d", range(50, 1000, 50))
@pytest.mark.parametrize("s", range(10))
def test_qpqc_values_standard_ill_cond_2(d, s):
    # Standard
    np.random.seed(s)
    W = loguniform.rvs(a=1e-6, b=1e6, size=(d,))
    idx = np.random.choice(d, d//2)
    W[idx] = -1* W[idx]
    standard_problem(W)


if __name__ == "__main__":
    test_qpqc_values_standard_ill_cond_2(900, 9)