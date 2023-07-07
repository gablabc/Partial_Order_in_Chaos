# %%
""" Compute the Feature Attributions/Importance for Spline models """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, t, levene

# Local imports
from utils import setup_pyplot_font, load_spline_model
from data_utils import DATASET_MAPPING

import sys, os
sys.path.append(os.path.join('../'))
from uxai.plots import bar

setup_pyplot_font(20)
save_path = os.path.join('Images', 'Kaggle-Houses')

# %%[markdown]
## Load Data and Model
# %%

# remove_correlations = False
remove_correlations = True

# Get the data
X, y, features, _ = DATASET_MAPPING["kaggle_houses"](remove_correlations)
N = X.shape[0]

# Load the model and spline parameters
model, simple_feature_idx, complex_feature_idx, degree, n_knots = \
                                load_spline_model(remove_correlations)
dim = n_knots + degree - 2

# Reorder feature names since we apply a SplineTransform before the model
reorder_feature_names = [features.names[i] for i in simple_feature_idx] +\
                        [features.names[i] for i in complex_feature_idx]

# Transform the data
H = model[0].transform(X)

# %% Ensure that we have the correct number of dimensions

assert dim * len(complex_feature_idx) + len(simple_feature_idx) == model[1].n_features
idxs =[[i] for i in range(len(simple_feature_idx))]
idxs += [list(range(i, i+dim)) for i in range(len(simple_feature_idx), model[1].n_features, dim)]
print(idxs)
assert len(idxs) == len(features)

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

# %%[markdown]
## Local Feature Attributions
# %%

# Compute LFA on the whole data
rashomon_po = model[1].feature_attributions(H, idxs=idxs)

# Explore range of tolerance for sensitivity analysis
step = (chosen_tolerance - model[1].RMSE - 1e-6)/10
tolerance_space = step * np.arange(0, 20) + model[1].RMSE + 1e-6
epsilon_space = model[1].get_epsilon(tolerance_space)
cardinalities = rashomon_po.get_utility(epsilon_space)
median_cardinalities = np.median(cardinalities, 0)
cardinalities = cardinalities[::20]

# %% Plot the Cardinality vs epsilon

fig, ax = plt.subplots()
jitter_x = np.random.uniform(-1e-4, 1e-4, size=cardinalities.shape)
jitter_y = np.random.uniform(-1e-3, 1e-3, size=cardinalities.shape)
ax.plot(tolerance_space, median_cardinalities, 'b-o')
ax.scatter(tolerance_space+jitter_x, cardinalities+jitter_y, alpha=0.3, s=1)
ax.plot(chosen_tolerance * np.ones(2), [0, 0.19], 'k--')
ax.text(chosen_tolerance, 0.2, r'$\epsilon$', horizontalalignment='center')
ax.set_xlabel('RMSE Tolerance')
ax.set_ylabel('Cardinality')
ax.set_xlim(model[1].RMSE, tolerance_space.max())
ax.set_ylim(0, 0.5)
plt.savefig(os.path.join(save_path, 
            f"AR_remove_corr_{remove_correlations}.pdf"), bbox_inches='tight')
plt.show()

# %% Local Feature Attributions of houses with high/low prices

sorted_price_idx = np.argsort(y.ravel())
# top 10 least/most expensive houses
idxs_to_explain = np.concatenate((sorted_price_idx[:10], 
                                  sorted_price_idx[-10:]))

# Return the partial order
extreme_attribs = rashomon_po.minmax_attrib(chosen_epsilon)

for i in idxs_to_explain:
    print(f"### Instance {i} ###\n")

    x_study = X[i]
    x_map = features.map_values(x_study)
    # Map to the transformation h
    h_map = [x_map[i] for i in simple_feature_idx] +\
            [x_map[i] for i in complex_feature_idx]
    print(h_map)

    # Predictions
    pred_mean, minmax_preds = model[1].predict(H[[i]], chosen_epsilon)
    pred_mean = np.exp(pred_mean)
    minmax_preds = np.exp(minmax_preds)
    print(f"True Price {np.exp(y[i, 0]):.0f} $")
    print(f"Point Prediction {pred_mean[0, 0]:.0f} $")
    print(f"Predictions {minmax_preds[0, 0]:.0f}, {minmax_preds[0, 1]:.0f} $")

    # Local Feature Attribution
    PO = rashomon_po.get_poset(i, chosen_epsilon, h_map)
    # Gap is well defined?
    if PO is not None:
        dot = PO.print_hasse_diagram(show_ambiguous=False)
        dot.render(filename=os.path.join(save_path, "PO", 
                    f"PO_instance_{i}_remove_corr_{remove_correlations}"), format='pdf')

        # Bar plot
        widths = (extreme_attribs[i, :, 1] - extreme_attribs[i, :, 0])/2
        bar(PO.phi_mean, h_map, xerr=widths.T)
        plt.savefig(os.path.join(save_path, "PO", 
                    f"Attrib_instance_{i}_remove_corr_{remove_correlations}.pdf"), 
                    bbox_inches='tight')
        plt.close()
    else:
        print("Gaps is not well-defined")
    print("\n")

# %%[markdown]
## Global Feature Importance
# %%

# Compute the GFI
min_max_importance, PO = model[1].feature_importance(chosen_epsilon,
                                feature_names=reorder_feature_names,
                                idxs=idxs, threshold=0.001, top_bottom=True)

# Bar plot
widths = np.abs(min_max_importance - PO.phi_mean.reshape((-1, 1)))
bar(PO.phi_mean, reorder_feature_names, xerr=widths.T)
plt.savefig(os.path.join(save_path, "PO", 
            f"Global_Importance_remove_corr_{remove_correlations}.pdf"), 
            bbox_inches='tight')

# Print PO
dot = PO.print_hasse_diagram(show_ambiguous=False)
filename = f"PO_Global_Importance_remove_corr_{remove_correlations}"
dot.render(os.path.join(save_path, "PO", filename), format='pdf')
print("\n")

plt.show()
# %%[markdown]
## Ill-defined Gaps
# %%

defined_gaps = rashomon_po.gap_crit_eps >= chosen_epsilon
ratio_defined_gaps = np.mean(defined_gaps)
print(f"Gap is well-defined on {100* ratio_defined_gaps:.1f} of the data")

# %% Investigate which instances do not have a well-defined gap

plt.figure()
_, _, rects = plt.hist(all_preds[~defined_gaps], bins=50, color='blue', 
                       alpha=0.3, density=True, label="Ill-defined")
plt.hist(all_preds[defined_gaps], bins=50, color='red', alpha=0.3, 
                density=True, label="Well-defined")
max_height = 0.97* max([h.get_height() for h in rects])
plt.plot(all_preds.mean() * np.ones(2), [0, max_height], 'k-')
plt.text(all_preds.mean(), max_height, 
         r"$\mathbb{E}_{\bm{z}\sim\mathcal{B}}[h_S(\bm{z})]$", 
         horizontalalignment='left')
plt.xlim(all_preds.min(), all_preds.max())
plt.xlabel("Predictions")
plt.ylabel("Probability Density")
plt.legend()
plt.savefig(os.path.join(save_path, 
            f"Gap_remove_corr_{remove_correlations}.pdf"), bbox_inches='tight')
plt.show()

# %% Define another background for ill-defined instances

# Compute LFA using cheap houses as the background
background = H[all_preds < np.quantile(all_preds, 0.25)]
rashomon_po = model[1].feature_attributions(H[~defined_gaps], 
                                            background=background,
                                            idxs=idxs)

defined_gaps_new = rashomon_po.gap_crit_eps >= chosen_epsilon
ratio_defined_gaps = np.mean(defined_gaps_new)
print(f"Gap is well-defined on {100* ratio_defined_gaps:.1f} of the foreground")


# %%
