import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join('../..'))

from uxai.utils_optim import opt_qpqc
from uxai.linear import get_ellipse_border


def plot_ellipsoid(A_half_inv, x_hat):
    zz = get_ellipse_border(A_half_inv, x_hat)
    plt.plot(zz[0, :], zz[1, :], 'k--', alpha = 0.5)
    plt.plot(x_hat[0], x_hat[1], 'kx')


def plot_plane(xmin, xmax, a, b):
    xx = np.linspace(xmin, xmax, 2)
    yy = -1 * (a[0] * xx - b) / a[1]
    plt.plot(xx, yy, 'k')


A_1 = np.array([[1.0, 0],
                [0, 1.0]])
x_hat_1 = np.array([0, 0]).reshape((-1, 1))

### Verification ###
xx, yy = np.meshgrid(np.linspace(-1.25, 1.25, 100), 
                     np.linspace(-1.25, 1.25, 100))


def test():
    def f(X):
        X_tilde = X - x_hat_2.T
        return np.sum(X_tilde.dot(A_2) * X_tilde, axis=1, keepdims=True)

    min_val, argmin, max_val, argmax = opt_qpqc(A_2, x_hat_2)
    print(f"Minimum value : {min_val}")
    print(f"Maximum value : {max_val}", "\n")

    plt.figure(figsize=(5, 7))
    ff = f(np.column_stack( (xx.ravel(), yy.ravel()) )).reshape(xx.shape)
    plot_ellipsoid(A_1, x_hat_1)
    CS = plt.contour(xx, yy, ff, levels=30)
    plt.plot(argmin[0, 0], argmin[1, 0], "r*")
    plt.plot(argmax[0, 0], argmax[1, 0], "r*")
    plt.clabel(CS, inline=True, fontsize=10)
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    # plt.axis('equal')




############# Visual Tests ###############

# Test 1
print("Almost-Axis Aligned Ellipsoid : Extern")
x_hat_2 = np.array([2, 0.01]).reshape((-1, 1))
A_2 = np.array([[1.0, 0],
                [0, 4.0]])
test()

# # Test 2
print("Non-Axis Aligned Ellipsoid : Extern")
x_hat_2 = np.array([2, 1.5]).reshape((-1, 1))
V = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
A_2 = V.T.dot(A_2.dot(V))
test()

# Test 3
print("Almost-Axis Aligned Ellipsoid : Intern")
x_hat_2 = np.array([0.75, 0.01]).reshape((-1, 1))
A_2 = np.array([[1.0, 0],
                [0, 4.0]])
test()

# Test 4
print("Non-Axis Aligned Ellipsoid : Intern")
x_hat_2 = np.array([0.75, 0.33]).reshape((-1, 1))
V = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
A_2 = V.T.dot(A_2.dot(V))
test()

# Test 5
print("Axis Aligned Hyperbolic-Paraboloid : Extern")
x_hat_2 = np.array([2, 0.01]).reshape((-1, 1))
A_2 = np.array([[-1.0, 0],
                [0, 4.0]])
test()

# Test 6
print("Non-Axis Aligned Hyperbolic-Paraboloid : Extern")
x_hat_2 = np.array([2, 1.5]).reshape((-1, 1))
V = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
A_2 = V.T.dot(A_2.dot(V))
test()

# Test 7
print("Axis Aligned Hyperbolic-Paraboloid : Intern")
x_hat_2 = np.array([0.75, 0.01]).reshape((-1, 1))
A_2 = np.array([[-1.0, 0],
                [0, 4.0]])
test()

# Test 8
print("Non-Axis Aligned Hyperbolic-Paraboloid : Intern")
x_hat_2 = np.array([0.75, 0.33]).reshape((-1, 1))
V = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
A_2 = V.T.dot(A_2.dot(V))
test()


plt.show()
