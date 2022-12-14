# %%

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc
rc('font',**{'family':'serif', 'serif':['Computer Modern Roman'], 'size':11})
rc('text', usetex=True)
from matplotlib import rcParams
rcParams['text.latex.preamble']=r"\usepackage{bm}\usepackage{amsfonts}"

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import math, os, sys
sys.path.append(os.path.join('../..'))

from uxai.kernels import KernelRashomon

# %% Setup

np.random.seed(42)
sigma_noise = 0.5
M = 600

Theta = np.random.uniform(2*math.pi, 4.5*math.pi, size=(M,))

X1 = Theta * np.cos(Theta) + sigma_noise * np.random.normal(size=(M,))
X2 = Theta * np.sin(Theta) + sigma_noise * np.random.normal(size=(M,))

y = 0.5 * ( Theta * np.sqrt(1 + Theta**2) + np.arcsinh(Theta) )
X = np.column_stack((X1, X2))

X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y.reshape((-1, 1)))

# %%
plt.figure()
train_plot = plt.scatter(X[:, 0], X[:, 1], s=10, c=y.ravel())
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
cbar = plt.colorbar(train_plot)
cbar.ax.set_ylabel(r"Target $y$", labelpad=-40, y=1.05, rotation=0)
plt.axis('equal')
plt.show()

# %%

train_size = 500
# kr = GridSearchCV(
#     KernelRashomon(kernel="rbf", gamma=0.1),
#     param_grid={"lambd": np.logspace(-3, 2, 10), "gamma": np.logspace(-3, 1, 10)},
kr = GridSearchCV(
    KernelRashomon(kernel="poly", gamma=0.1, degree=3),
    param_grid={"lambd": np.logspace(-3, 2, 10), "gamma": np.logspace(-3, 1, 10)},
)
kr.fit(X[:train_size], y[:train_size])
kr = kr.best_estimator_

# %% Utility functions

def plot_value_var(models, uncertainty=True, n_points=300):

    n = 100
    XX, YY = np.meshgrid(np.linspace(-1.8, 1.8, n), 
                         np.linspace(-1.8, 1.8, n))
    XX_ = np.column_stack((XX.ravel(), YY.ravel()))

    preds, min_max_preds = models.predict(XX_, epsilon=0.05)

    plt.figure()
    if uncertainty:
        varmap = plt.contourf(XX, YY, np.diff(min_max_preds,axis=-1).reshape(XX.shape), 
                                            cmap='Blues', alpha=0.75)
    else:
        varmap = plt.contourf(XX, YY, preds.reshape(XX.shape), 
                                            cmap='Blues', alpha=0.75)
    plt.scatter(X[:, 0], X[:, 1], c='k', s=8)
    plt.axis('scaled')
    plt.xlim(-1.8, 1.8)
    plt.ylim(-1.8, 1.8)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    cbar = plt.colorbar(varmap,fraction=0.046, pad=0.04)

plot_value_var(kr)
plot_value_var(kr, uncertainty=False)
plt.show()

# %%
from scipy.stats import chi2
# Sample points in the Rashomon set
rel_epsilon = 0.1
abs_epsilon = kr.get_epsilon(rel_epsilon)
z = np.random.normal(0, 1, size=(100000, kr.R)) / np.sqrt(chi2.ppf(0.5, df=kr.R))
z = z[np.linalg.norm(z, axis=1) <= 1]
alpha_inside = z.dot(kr.A_half_inv).T + kr.alpha_s

# %%
XX, YY = np.meshgrid(np.linspace(-2, 2, 10), 
                     np.linspace(-2, 2, 10))
XX_ = np.column_stack((XX.ravel(), YY.ravel()))

print(XX_.shape)
grad_ = kr.gradients(XX_)

plt.figure(figsize = (10, 10))

for i in range(XX_.shape[0]):
    for j in np.random.choice(alpha_inside.shape[1], size=(20)):
        alpha = alpha_inside[:, [j]].T
        grad = np.sum(grad_[i,...]*alpha, axis=1)
        # normalize gradient
        norm = np.sqrt(grad[0] ** 2 + grad[1] ** 2) * 2
        plt.quiver(XX_[i, 0], XX_[i, 1],
                   grad[0], grad[1], 
                   units='xy', scale=2, alpha=0.1, zorder=0)
plt.axis('equal')

plt.scatter(X[:, 0], X[:, 1], zorder=-1)
plt.show()

# %%

## Local Feature Attributions ##
z = np.array([[0.5, 0]])
X_explain = X[:5]

plt.figure()
plt.scatter(X[:, 0], X[:, 1],  s=8, c=y.ravel())
for i in range(5):
    plt.text(X[i, 0], X[i, 1], str(i), c='r', size=20)
plt.axis('scaled')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.show()

# %%
from IPython import display

rashomon_po = kr.attributions(X_explain, z)

# %%
extreme_attribs = rashomon_po.minmax_attrib(abs_epsilon)
ensemble_attribs = np.sum(kr.IG[...,np.newaxis] * alpha_inside.reshape((1, 1, kr.R, -1)), axis=2)

# %%
for i in range(5):
    print(f"#### {i} ####")
    plt.figure()
    extreme_attrib = extreme_attribs[i]
    plt.plot(range(1, 3), ensemble_attribs[i, :, ::10000], 'b')
    plt.plot(range(1, 3), extreme_attrib[:, 0], 'r')
    plt.plot(range(1, 3), extreme_attrib[:, 1], 'r')
    plt.plot(range(1, 3), [0, 0], 'k')
    plt.show()
    local_po = rashomon_po.get_poset(i, abs_epsilon, ["x1", "x2"])
    if local_po is not None:
        dot = local_po.print_hasse_diagram()
        display.display_svg(dot)
    else:
        print("No defined gap")
# dot.render(filename=os.path.join('Images', 'PO_Lobal_Spline'), format='png')

# plt.show()


# %%
