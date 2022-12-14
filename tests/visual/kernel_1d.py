import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from scipy.stats import chi2

import sys, os
sys.path.append(os.path.join('../..'))
from uxai.kernels import KernelRashomon


np.random.seed(42)
X = np.random.uniform(-2, 2, size=(5000, 1))
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += np.random.uniform(-1.5, 1.5, size=(X.shape[0] // 5,))
X_plot = np.linspace(-3, 3, 1000)[:, None]

train_size = 100

### Automated ###
# kr = GridSearchCV(
#     KernelRashomon(kernel="rbf", gamma=0.1),
#     param_grid={"lambd": np.logspace(-3, 2, 10), "gamma": np.logspace(-3, 1, 10)},
# )
kr = GridSearchCV(
    KernelRashomon(kernel="poly", gamma=0.1),
    param_grid={"lambd": np.logspace(-3, 2, 10), "gamma": np.logspace(-3, 1, 10)},
)
kr.fit(X[:train_size], y[:train_size])
kr = kr.best_estimator_
lambd = kr.lambd

### Manual ###
# lambd = 1e-2
# gamma = 20
# kr = KernelRashomon(kernel="rbf", gamma=20, lambd=lambd)
# kr = KernelRashomon(kernel="poly", gamma=gamma, lambd=lambd, degree=3)
# kr.fit(X[:train_size], y[:train_size])

# Assert Performance
print(f"Train loss = {kr.MSE:.4f} + {lambd} {kr.h_norm():.4f} = {kr.train_loss:.4f}")
epsilon_rel = 0.05
upper_bound_loss = (1 + epsilon_rel) * kr.train_loss
print(f"Upper bound (1 + epsilon') (L(a) + lambda |h_a|^2) : {upper_bound_loss:.4f}")
# Get the absolute epsilon
epsilon = kr.get_epsilon(epsilon_rel)

# Sample points in the Rashomon set
z = np.random.normal(0, 1, size=(100000, kr.R)) / np.sqrt(chi2.ppf(0.2, df=kr.R))
z = z[np.linalg.norm(z, axis=1) <= 1]
alpha_inside = z.dot(kr.A_half_inv).T * np.sqrt(epsilon) + kr.alpha_s # (R, Nsamples)
all_preds = np.dot(kr.get_kernel(X_plot, kr.Dict), alpha_inside) + kr.mu

# Ensure that all models have loss bellow epsilon
K_inside = kr.get_kernel(kr.Dict)
all_MSE = np.mean((y[:train_size].reshape((-1, 1)) - K_inside.dot(alpha_inside) - kr.mu) ** 2, axis=0)
all_h_norms = np.sum(alpha_inside * K_inside.dot(alpha_inside), axis=0)
all_losses = all_MSE + lambd * all_h_norms
print(f"Ensemble : {np.min(all_losses):.4f}, {np.max(all_losses):.4f}")

# Plot Prediction Uncertainty
y_kr, min_max_preds = kr.predict(X_plot, epsilon=epsilon)

plt.figure()
plt.scatter(X[:200], y[:200], c="k", label="data", zorder=1, edgecolors=(0, 0, 0))
plt.plot(X_plot, y_kr, c="g")
plt.plot(X_plot, all_preds[:, ::100], 'r', alpha=0.2)
plt.fill_between(X_plot.ravel(), min_max_preds[:, 0], min_max_preds[:, 1], alpha=0.25)
plt.xlim(-3, 3)
plt.ylim(y.min()-0.1, y.max()+0.1)
plt.xlabel("data")
plt.ylabel("target")
plt.show()
