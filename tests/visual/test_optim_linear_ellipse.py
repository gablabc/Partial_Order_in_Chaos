"""
Test the optimization of a linear
function over an ellipsoid.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join('../..'))

from uxai.utils import Ellipsoid


def plot_ellipsoid(ellipsoid):
    zz = ellipsoid.get_ellipse_border(2)
    plt.plot(zz[0, :], zz[1, :], 'k--', alpha = 0.5)
    plt.plot(x_hat[0], x_hat[1], 'kx')


# Generate ellipsoid object
x_hat = np.array([1., 0.5]).reshape((-1, 1))
A_half_inv = np.array([[ 1.0, 2.0],
                       [-0.5, 1.0]])
A = A_half_inv.dot(A_half_inv.T)
ellipsoid = Ellipsoid(A, x_hat)


### Verification ###
xx, yy = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))

# Test 1
a = np.array([[1], [1]])
ff = a[0] * xx + a[1] * yy

plt.figure()
plot_ellipsoid(ellipsoid)
CS = plt.contour(xx, yy, ff)
plt.clabel(CS, inline=True, fontsize=10)
minmax_val, minmax_sol = ellipsoid.opt_linear(a, epsilon=2, return_input_sol=True)
print(minmax_val)
plt.plot(minmax_sol[:, 0, 0], minmax_sol[:, 0, 1], 'ko')
plt.xlim(-4, 4)
plt.ylim(-4, 4)


# # Test 2
a[1] = a[1] * -1
ff = a[0] * xx + a[1] * yy

plt.figure()
plot_ellipsoid(ellipsoid)
CS = plt.contour(xx, yy, ff)
plt.clabel(CS, inline=True, fontsize=10)
minmax_val, minmax_sol = ellipsoid.opt_linear(a, epsilon=2, return_input_sol=True)
print(minmax_val)
plt.plot(minmax_sol[:, 0, 0], minmax_sol[:, 0, 1], 'ko')
plt.xlim(-4, 4)
plt.ylim(-4, 4)


# # Test 3
a = np.array([[0], [3]])
b = 0
ff = a[0] * xx + a[1] * yy

plt.figure()
plot_ellipsoid(ellipsoid)
CS = plt.contour(xx, yy, ff)
plt.clabel(CS, inline=True, fontsize=10)
minmax_val, minmax_sol = ellipsoid.opt_linear(a, epsilon=2, return_input_sol=True)
print(minmax_val)
plt.plot(minmax_sol[:, 0, 0], minmax_sol[:, 0, 1], 'ko')
plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.show()