import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

import sys, os
sys.path.append(os.path.join('../..'))

from uxai.linear import LinearRashomon

np.random.seed(42)

# Generate Data
def generate_data(n_samples, noise):
    n_samples = 1000
    Z = np.random.normal(size=(n_samples, 2))
    X = Z.dot(np.array([[1, 2], [-0.5, 0.25]]))
    y = 2 * X[:, [0]] + X[:, [1]] + noise * np.random.normal(size=(n_samples,1))
    return X, y

noise = 0.2
X, y = generate_data(2000, noise)

linear_rashomon = LinearRashomon().fit(X, y)
RMSE = linear_rashomon.RMSE[0]
print(f"RMSE : {RMSE}")
print(f"Maxmimum tolerable RMSE : {RMSE + 0.05}")
epsilon = linear_rashomon.get_epsilon(RMSE + 0.05)

# Sample points in the Rashomon set Boundary
z = np.random.normal(0, 1, size=(40000, 3))
z = z / np.linalg.norm(z, axis=1, keepdims=True)
w_boundary = z.dot(linear_rashomon.A_half_inv) * np.sqrt(epsilon) + linear_rashomon.w_hat.T


# Is the prediction method working?
predict_method = linear_rashomon.predict(X)
X_ = (X - linear_rashomon.X_mean) / linear_rashomon.X_std
X_tilde = np.column_stack( (np.ones(len(X_)), X_) )
manual_pred =  linear_rashomon.y_std * X_tilde.dot(linear_rashomon.w_hat) + linear_rashomon.y_mean
assert np.isclose(predict_method, manual_pred).all()
assert linear_rashomon.get_RMSE(X, y) == RMSE


# Ensure that all models have RMSE bellow epsilon
all_RMSE = np.sqrt(np.mean((y - linear_rashomon.y_std * X_tilde.dot(w_boundary.T) - linear_rashomon.y_mean) ** 2, axis=0))
print(np.min(all_RMSE), np.max(all_RMSE))


# Plot the Rashomon Set
plt.figure()
scale = linear_rashomon.y_std / linear_rashomon.X_std
linear_rashomon.plot_rashomon_set(0, 1, epsilon)
plt.scatter(scale[0] * w_boundary[::4, 1], scale[1] * w_boundary[::4, 2], c="r", marker="x")
plt.xlabel(r"$w_1$")
plt.ylabel(r"$w_2$")


# Show extreme values of w
print(linear_rashomon.min_max_coeffs(epsilon))


line = np.linspace(-3, 3, 100)
# Show PDP
for idx in [0, 1]:

    pdp = linear_rashomon.partial_dependence(line, idx=idx, epsilon=epsilon)
    plt.figure()
    plt.fill_between(line, pdp[:, 0], pdp[:, 1], alpha=0.5)
    plt.xlabel(f"$w{idx}$")
    plt.ylabel("Partial Dependence")


plt.figure()
# Plot the GFI and compare with sampling
bp = plt.boxplot(w_boundary[:, 1:] * y.std(), patch_artist=True)

for patch in bp['boxes']:
    patch.set_facecolor('#9999FF')

extreme_imp, global_po = linear_rashomon.feature_importance(epsilon, ["x1", "x2"])
plt.plot(range(1, 3), extreme_imp[:, 0], 'r')
plt.plot(range(1, 3), extreme_imp[:, 1], 'r')


dot = global_po.print_hasse_diagram()
dot.render(filename=os.path.join('Images', 'PO_Global_Linear'), format='png')


# Plot the LFA and compare with sampling

plt.figure()
bp = plt.boxplot(w_boundary[:, 1:] * y.std() / X.std(0) * (X[0] - np.mean(X, 0)), patch_artist=True)

for patch in bp['boxes']:
    patch.set_facecolor('#9999FF')

rashomon_po = linear_rashomon.attributions(X)
extreme_attribs = rashomon_po.minmax_attrib(epsilon)[0]
plt.plot(range(1, 3), extreme_attribs[:, 0], 'r')
plt.plot(range(1, 3), extreme_attribs[:, 1], 'r')

# Partial order
local_po = rashomon_po.get_poset(0, epsilon, ["x1", "x2"])
dot = local_po.print_hasse_diagram()
dot.render(filename=os.path.join('Images', 'PO_Lobal_Linear'), format='png')

# Plot utility as a function of epsilon
tolerance = RMSE + np.linspace(0, 0.5, 200)
epsilon_space = linear_rashomon.get_epsilon(tolerance)
utility = rashomon_po.get_utility(epsilon_space)
confidence = chi2(df=X.shape[0]).cdf(X.shape[0]*tolerance**2/noise**2)

plt.figure()
plt.plot(utility, confidence, 'k')
plt.xlabel('Utility')
plt.ylabel('RMSE tolerance')

# Plot Prediction Uncertainty
xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
zz = np.column_stack( (xx.ravel(), yy.ravel()) )
_, preds = linear_rashomon.predict(zz, epsilon)
uncertainty = np.diff(preds, axis=1).reshape(xx.shape)

plt.figure()
varmap = plt.contourf(xx, yy, uncertainty, cmap='Blues', alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.colorbar(varmap, fraction=0.046, pad=0.04)

plt.show()
