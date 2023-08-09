# %%
""" Compute the Feature Attributions/Importance for Spline models """

import numpy as np
import matplotlib.pyplot as plt

# Local imports
from utils import setup_pyplot_font, load_spline_model, get_grouped_features
from data_utils import DATASET_MAPPING

import sys, os
sys.path.append(os.path.join('../'))
from uxai.plots import bar

setup_pyplot_font(20)
save_path = os.path.join('Images', 'Kaggle-Houses')

# %%[markdown]
## Load Data and Model
# %%

# Get the data
X, y, features, _ = DATASET_MAPPING["kaggle_houses"](remove_correlations=False)
N, d = X.shape
# Get groups of correlated features
correlated_features = get_grouped_features(X, features.names)

# Load the model and spline parameters
model, simple_feature_idx, complex_feature_idx, degree, n_knots = \
                                load_spline_model(remove_correlations=False)
dim = n_knots + degree - 2

# Transform the data
H = model[0].transform(X)

# Sanity check
assert dim * len(complex_feature_idx) + len(simple_feature_idx) == model[1].n_features

# %% Set epsilon based on the previous script
chosen_tolerance = 0.1444
chosen_epsilon = model[1].get_epsilon(chosen_tolerance)
print(f"Model RMSE : {model[1].RMSE:.4f}")
print(f"Tolerance RMSE : {chosen_tolerance:.4f}")

# Space of alternative tolerances to explore
step = (chosen_tolerance - model[1].RMSE - 1e-6)/10
tolerance_space = step * np.arange(0, 20) + model[1].RMSE + 1e-6
epsilon_space = model[1].get_epsilon(tolerance_space)

# %% Map each feature to its corresponding columns of H
start = len(simple_feature_idx)
idxs_non_grouped = []
for i in range(len(features)):
    if i in simple_feature_idx:
        idxs_non_grouped.append([simple_feature_idx.index(i)])
    else:
        j = complex_feature_idx.index(i)
        idxs_non_grouped.append(list(range(start+j*dim, start+(j+1)*dim)))
print(idxs_non_grouped)

# %% Group features to account to Spline bases AND Correlations
list_correl_features = np.ravel(list(correlated_features.values()))
grouped_feature_names = []
idxs_grouped = []
for i in range(len(features)):
    if i not in list_correl_features:
        idxs_grouped.append(idxs_non_grouped[i])
        grouped_feature_names.append(features.names[i])
for key, value in correlated_features.items():
    idxs_grouped.append(idxs_non_grouped[value[0]] + \
                        idxs_non_grouped[value[1]])
    grouped_feature_names.append(key)
print(idxs_grouped)

# %%[markdown]
## Local Feature Attributions
# %% Local Feature Attributions of houses with high/low prices

sorted_price_idx = np.argsort(y.ravel())
# top 10 least/most expensive houses
idxs_to_explain = np.concatenate((sorted_price_idx[:10],
                                  sorted_price_idx[-10:]))

# %%
# Sample LFA for correlated features and scatter plot
lstsq_phis, sampled_phis = model[1].sample_feature_attributions(
                                    H[idxs_to_explain], chosen_epsilon, 
                                    idxs=idxs_non_grouped, n_samples=500)

# %%
for i, orig_index in enumerate(idxs_to_explain):
    print(f"### Instance {orig_index} ###\n")
    for (j, k) in correlated_features.values():
        plt.figure()
        plt.scatter(sampled_phis[i, j], 
                    sampled_phis[i, k], alpha=0.1)
        plt.plot(lstsq_phis[i, j], lstsq_phis[i, k], "r*")
        max_width = np.max(np.abs(sampled_phis[i, [j, k]]))
        plt.plot(1.2*max_width * np.array([-1, 1]), np.zeros(2), 'k-')
        plt.plot(np.zeros(2), 1.2*max_width * np.array([-1, 1]), 'k-')
        plt.xlabel(f"Attribution of {features.names[j]}")
        plt.ylabel(f"Attribution of {features.names[k]}")
        plt.axis('square')
        plt.xlim(-max_width, max_width)
        plt.ylim(-max_width, max_width)
        plt.savefig(os.path.join(save_path, "PO", 
                        f"Correls_{j}_{k}_instance_{orig_index}.pdf"), 
                        bbox_inches='tight')
        plt.close()


# %% Compute LFA on the whole data

group_features = True

if not group_features:
    idxs = idxs_non_grouped
    feature_names = features.names
else:
    idxs = idxs_grouped
    feature_names = grouped_feature_names

# LFA
rashomon_po = model[1].feature_attributions(H, idxs=idxs)
extreme_attribs = rashomon_po.minmax_attrib(chosen_epsilon)
cardinalities = rashomon_po.get_utility(epsilon_space)
median_cardinalities = np.median(cardinalities, 0)

# %% Plot the Cardinality vs epsilon

# plt.plot(tolerance_space, median_cardinalities, 'r-o', label="No Grouping")
plt.plot(tolerance_space, median_cardinalities, 'b-o', label="Grouping")
plt.plot(chosen_tolerance * np.ones(2), [0, 0.19], 'k--')
plt.text(chosen_tolerance, 0.2, r'$\epsilon$', horizontalalignment='center')
plt.xlabel('RMSE Tolerance')
plt.ylabel('Cardinality')
plt.xlim(model[1].RMSE, tolerance_space.max())
plt.ylim(0, 0.5)
plt.legend()
# plt.savefig(os.path.join(save_path, f"AR_remove_corr_False.pdf"), bbox_inches='tight')
plt.show()

# %% Local Feature Attributions of houses with high/low prices

for i in idxs_to_explain:
    print(f"### Instance {i} ###\n")

    x_study = X[i]
    x_map = features.map_values(x_study)
    
    #If correlated features are grouped
    if group_features:
        x_map_p = []
        for feature_idx in range(len(features)):
            if feature_idx not in list_correl_features:
                x_map_p.append(x_map[feature_idx])
        x_map_p += list(correlated_features.keys())
        x_map = x_map_p

    print(x_map)

    # Predictions
    pred_mean, minmax_preds = model[1].predict(H[[i]], chosen_epsilon)
    pred_mean = np.exp(pred_mean)
    minmax_preds = np.exp(minmax_preds)
    print(f"True Price {np.exp(y[i, 0]):.0f} $")
    print(f"Point Prediction {pred_mean[0, 0]:.0f} $")
    print(f"Predictions {minmax_preds[0, 0]:.0f}, {minmax_preds[0, 1]:.0f} $")

    # Local Feature Attribution
    PO = rashomon_po.get_poset(i, chosen_epsilon, x_map)
    # Gap is well defined?
    if PO is not None:

        dot = PO.print_hasse_diagram(show_ambiguous=False)
        dot.render(filename=os.path.join(save_path, "PO", 
                    f"PO_instance_{i}_group_{group_features}"), format='pdf')

        # Bar plot
        widths = (extreme_attribs[i, :, 1] - extreme_attribs[i, :, 0])/2
        bar(PO.phi_mean, x_map, xerr=widths.T)
        plt.savefig(os.path.join(save_path, "PO", 
                    f"Attrib_instance_{i}_group_{group_features}.pdf"), 
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
                                feature_names=feature_names,
                                idxs=idxs, 
                                threshold=0.001, top_bottom=True)

# Bar plot
widths = np.abs(min_max_importance - PO.phi_mean.reshape((-1, 1)))
bar(PO.phi_mean, feature_names, xerr=widths.T)
plt.savefig(os.path.join(save_path, "PO",
            f"Global_Importance_group_{group_features}.pdf"), 
            bbox_inches='tight')

# Print PO
dot = PO.print_hasse_diagram(show_ambiguous=False)
filename = f"PO_Global_Importance_group_{group_features}"
dot.render(os.path.join(save_path, "PO", filename), format='pdf')
print("\n")

# plt.show()

# %%[markdown]
## Ill-defined Gaps
# %%

defined_gaps = rashomon_po.gap_crit_eps >= chosen_epsilon
ratio_defined_gaps = np.mean(defined_gaps)
print(f"Gap is well-defined on {100* ratio_defined_gaps:.1f} of the data")

# %% Investigate which instances do not have a well-defined gap

plt.figure()
all_preds = model.predict(X).ravel()
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
            f"Gap_remove_corr_False.pdf"), bbox_inches='tight')
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

