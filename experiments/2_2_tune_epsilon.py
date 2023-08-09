# %%
""" Compute the error tolerance for Splines fit on Houses """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, t, levene

# Local imports
from utils import setup_pyplot_font, load_spline_model
from data_utils import DATASET_MAPPING

import os

setup_pyplot_font(20)
save_path = os.path.join('Images', 'Kaggle-Houses')

# %%[markdown]
## Load Data and Model
# %%

remove_correlations = False
# remove_correlations = True

# Get the data
X, y, features, _ = DATASET_MAPPING["kaggle_houses"](remove_correlations)
N, d = X.shape

# Load the model and spline parameters
model, simple_feature_idx, complex_feature_idx, degree, n_knots = \
                                load_spline_model(remove_correlations)
dim = n_knots + degree - 2

# Reorder feature names since we apply a SplineTransform before the model
reorder_feature_names = [features.names[i] for i in simple_feature_idx] +\
                        [features.names[i] for i in complex_feature_idx]

# Transform the data
H = model[0].transform(X)

# Sanity check
assert dim * len(complex_feature_idx) + len(simple_feature_idx) == model[1].n_features


# %%[markdown]
## Residual Analysis
# %%

# Residuals
all_preds = model.predict(X).ravel()
Delta = y.ravel() - all_preds
Delta_max = 0.75

# Test of Homogeneity of the Residuals
y_hat_split = np.array_split(np.sort(all_preds), 3)
Delta_split = np.array_split(Delta[np.argsort(all_preds)], 3)
_, p_val = levene(*Delta_split, center="trimmed")
print(f"P-value for variance homogeneity {100*p_val:.2f} %")


plt.figure()
for i in range(3):
    quartiles = np.quantile(Delta_split[i], [0.25, 0.5, 0.75])
    plt.fill_between([y_hat_split[i].min(), y_hat_split[i].max()],
                      quartiles[0] * np.ones(2),
                      quartiles[2] * np.ones(2),
                      alpha=0.5, color='grey')
    plt.plot([y_hat_split[i].min(), y_hat_split[i].max()], 
             quartiles[1] * np.ones(2), 'k-')
    plt.scatter(y_hat_split[i][::3], Delta_split[i][::3], alpha=0.2, c='b')
plt.xlabel(r"Prediction $h_S(\bm{x}^{(i)})$")
plt.ylabel(r'Residual $y^{(i)} - h_S(\bm{x}^{(i)})$')
plt.xlim(all_preds[::2].min(), all_preds[::2].max())
plt.ylim(-0.6, 0.6)
plt.savefig(os.path.join(save_path, 
            f"Homogeneity_remove_corr_{remove_correlations}.pdf"), bbox_inches='tight')
plt.show()

# %%
# VC pseudo-dimension bound
d = H.shape[1] + 1
M = np.max(Delta**2)
extra_tolerance = 2 * M * np.sqrt(2 / N)
extra_tolerance *= np.sqrt(d*np.log(2*np.e*N/d) + np.log(8/0.05))
chosen_tolerance = np.sqrt(model[1].MSE + extra_tolerance) 
print(f"Model RMSE : {model[1].RMSE:.4f}")
print(f"VC Tolerance RMSE : {chosen_tolerance:.4f}")

# %%
# Fit the residual density
line = np.linspace(-Delta_max, Delta_max, 1000)
params_norm = norm.fit(Delta, floc=0)
norm_distr = norm(*params_norm)
params_t = t.fit(Delta, floc=0)
t_distr = t(*params_t)
plt.figure()
plt.hist(Delta, bins=50, color='blue', alpha=0.3, density=True)
plt.plot(line, norm_distr.pdf(line), label="Normal")
plt.plot(line, t_distr.pdf(line), label="Student-t")
plt.xlabel(r'Residual $y^{(i)} - h_S(\bm{x}^{(i)})$')
plt.ylabel("Probability Density")
plt.xlim(-Delta_max, Delta_max)
plt.legend()
plt.savefig(os.path.join(save_path,
            f"Residual_remove_corr_{remove_correlations}.pdf"), bbox_inches='tight')
plt.show()

# %%
# Residuals have heavy tails So sample from a truncated t distribution
samples = t_distr.rvs(size=(int(2e5), X.shape[0]))
samples = np.clip(samples, a_min=-Delta_max, a_max=Delta_max)
synthetic_errors = np.mean(samples**2, axis=1)

# %%
# Use the empirical quantile of the simulated samples to set
# the tolerance in RMSE
chosen_tolerance = np.sqrt(np.quantile(synthetic_errors, 0.95))
chosen_epsilon = model[1].get_epsilon(chosen_tolerance)
print(f"Model RMSE : {model[1].RMSE:.4f}")
print(f"Tolerance RMSE : {chosen_tolerance:.4f}")

# %%
plt.figure()
_, _, rects = plt.hist(np.sqrt(synthetic_errors), bins=300,
                        color='blue', alpha=0.3, density=True)
max_height = 0.95 * max([h.get_height() for h in rects])
plt.plot(model[1].RMSE * np.ones(2), [0, max_height], 'k-')
plt.text(model[1].RMSE, max_height, "Least-Square", horizontalalignment='right')
plt.plot(chosen_tolerance * np.ones(2), [0, max_height], 'k-')
plt.text(chosen_tolerance, max_height, "Tolerance", horizontalalignment='left')
plt.xlabel('RMSE')
plt.xlim(0.1, 0.2)
plt.show()

# %%
