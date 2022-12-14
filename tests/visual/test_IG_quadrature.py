# %%

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':11})
rc('text', usetex=True)

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
import os, sys
sys.path.append(os.path.join('../..'))

from uxai.kernels import grad_rbf, grad_poly

# %% Setup

np.random.seed(42)
sigma_noise = 0.5
M = 600
r = np.array([[1, 0]])
x = np.array([[0.75, 0.25]])
z = np.array([[-1, -1]])

n = 100
XX, YY = np.meshgrid(np.linspace(-1.5, 1.5, n), 
                     np.linspace(-1.5, 1.5, n))
XX_ = np.column_stack((XX.ravel(), YY.ravel()))

# RBF
# metric="rbf"
# params = {"gamma" : 1}
# POLY"
metric = "poly"
params = {"gamma" : 0.5, "degree" : 3}

K = pairwise_kernels(XX_, r, metric=metric, **params)


t = np.linspace(0, 1, 100).reshape((-1, 1))
line = t * x + (1 - t) * z
K_line = pairwise_kernels(line, r, metric=metric, **params)

if metric == "rbf":
    grad = grad_rbf(line, r, K_line, **params)
elif metric == "poly":
    grad = grad_poly(line, r, **params)

varmap = plt.contourf(XX, YY, K.reshape(XX.shape), cmap='Blues', alpha=0.75)
plt.scatter(x[:, 0], x[:, 1], c='k', s=8)
plt.scatter(z[:, 0], z[:, 1], c='k', s=8)
plt.plot(line[:, 0], line[:, 1], 'k--')
plt.quiver(line[::5, 0], line[::5, 1] , 
            grad[::5, 0, 0]/2, grad[::5, 1, 0]/2, 
            units='xy', scale=2, zorder=1)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.show()

# %%

gap = K_line[-1][0] - K_line[0][0]
gap_error = []
print(f"Gap : {gap}")

steps = [5, 10, 25, 50, 100, 200, 400, 800]
for n in steps:
    delta_t = 1 / (n - 1)
    t = np.linspace(0, 1, n).reshape((-1, 1))
    line = t * x + (1 - t) * z
    K_line = pairwise_kernels(line, r, metric=metric, **params)
    if metric == "rbf":
        grad = grad_rbf(line, r, K_line, **params)[..., 0]
    elif metric == "poly":
        grad = grad_poly(line, r, **params)[..., 0]
    grad[0, :] /= 2
    grad[-1, :] /= 2
    EG = (x - z) * delta_t * grad.sum(0, keepdims=True)
    print(f"EG : {EG}")
    gap_error.append(np.abs(gap - EG.sum()))

    print(f"Gap Error : {gap_error[-1]}")

# %%

plt.figure()
plt.plot(steps, gap_error, "k-o")
plt.xscale('log')
plt.yscale('log')
plt.show()

# %%
