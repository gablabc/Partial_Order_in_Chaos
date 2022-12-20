# %%
""" Interactive code to analyse results of RFs on Adult-Income """
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from utils import custom_train_test_split, setup_pyplot_font, setup_data_trees

import sys, os
sys.path.append(os.path.join('../'))
from uxai.trees import all_tree_preds, epsilon_upper_bound
from uxai.plots import bar

setup_pyplot_font(20)

# %%[markdown]
# ## Load Data
# %%

X, y, features, task, ohe = setup_data_trees("adult_income")
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, task)
M = 1000
N = X_test.shape[0]

# %%[markdown]
# ## Plot Gap Errors
# %%

gap_errors = []
all_B = [10, 25, 50, 100, 500]
for seed in range(5):
    for B in all_B:
        tmp_filename = f"GapErrors_M_{M}_seed_{seed}_background_{B}.npy"
        gap_error_load = np.abs(np.load(os.path.join("models", "Adult-Income", tmp_filename)))
        gap_errors.append(np.column_stack((gap_error_load, seed * np.ones(2000), B * np.ones(2000))))
df = pd.DataFrame(np.vstack(gap_errors), columns=["Gap Errors", "Run", "Background Size"])


# %%

hue_order = list(range(5))
sns.boxplot(x="Background Size", y="Gap Errors", hue="Run", data=df, width=0.6, 
            order=all_B, hue_order=hue_order, showfliers=False)
plt.legend('', frameon=False)
plt.savefig(os.path.join("Images", "Adult-Income", f"RF_B_samples_M_{M}.pdf"), bbox_inches='tight', pad_inches=0)
# plt.show()


# %%
best_m_idx = 0
best_utility = 0
best_error = 0
best_seed = -1
for seed in range(5):
    model = load(os.path.join("models", "Adult-Income", f"RF_M_{M}_seed_{seed}.joblib"))

    # The upper bound on preformance
    tree_preds = all_tree_preds(ohe.transform(X_test), model, task="classification")
    m, epsilon_upper = epsilon_upper_bound(tree_preds, y_test.reshape((-1, 1)), 
                                            task="classification")

    # Load the pre-computed SHAP feature attributions
    tmp_filename = f"TreeSHAP_M_{M}_seed_{seed}_background_500.joblib"
    rashomon_po = load(os.path.join("models", "Adult-Income", tmp_filename))
    utility = rashomon_po.get_utility(epsilon_upper)
    utility /= np.max(utility)
    # true_score = 1 - model.score(ohe.transform(X_test), y_test)
    # confidence = 1 - np.exp(-0.5 * N * (epsilon_upper - true_score) ** 2)
    
    curr_utility_idx = np.argmax(epsilon_upper <= 0.17)
    if utility[curr_utility_idx] > best_utility:
        best_m_idx = curr_utility_idx
        best_utility = utility[curr_utility_idx]
        best_error = epsilon_upper[curr_utility_idx]
        best_seed = seed
    plt.plot(utility, epsilon_upper)

plt.plot(best_utility, best_error, 'r*', markersize=15, markeredgecolor='k')
plt.xlabel("Utility")
plt.xlim(0.35, 1)
plt.ylim(0.135, 0.22)
plt.ylabel(r"Test Error Bound $\epsilon^+$")
plt.savefig(os.path.join("Images", "Adult-Income", f"RF_utility_M_{M}.pdf"), bbox_inches='tight', pad_inches=0)
# plt.show()


# %%
# Select epsilon star and best seed
model = load(os.path.join("models", "Adult-Income", f"RF_M_{M}_seed_{best_seed}.joblib"))

# The upper bound on preformance
tree_preds = all_tree_preds(ohe.transform(X_test), model, task="classification")
m, epsilon_upper = epsilon_upper_bound(tree_preds, y_test.reshape((-1, 1)), 
                                        task="classification")
epsilon_star = epsilon_upper[best_m_idx]

# Shap feature attribution
tmp_filename = f"TreeSHAP_M_{M}_seed_{best_seed}_background_500.joblib"
rashomon_po = load(os.path.join("models", "Adult-Income", tmp_filename))

# %%[markdown]
# ## Show local feature attributions
# %%

preds = tree_preds.mean(1)
idx_no_capital = (X_test[:2000, 2]==0) & (X_test[:2000, 3]==0)
conf_pos = np.where(idx_no_capital & (preds[:2000]>0.75))[0]
conf_neg = np.where(idx_no_capital & (preds[:2000]<0.3) & (preds[:2000]>0.1))[0]

idxs = [conf_pos[0],
        conf_pos[1],
        conf_pos[2],
        conf_neg[0],
        conf_neg[1],
        conf_neg[2],
        42, 105, 225, 450, 451, 1560, 32, 345, 567, 1999, 1998, 1997]

descriptions = ["A confident prediction of y=1 with no Capital Gain/Loss",
                "Another confident prediction of y=1 with no Capital Gain/Loss",
                "Another confident prediction of y=1 with no Capital Gain/Loss",
                "A confident prediction of y=0 with no Capital Gain/Loss",
                "Another confident prediction of y=0 with no Capital Gain/Loss",
                "Another confident prediction of y=0 with no Capital Gain/Loss",
                "Random", "Random", "Random", "Random", "Random", "Random",
                "Random", "Random", "Random", "Random", "Random", "Random"]

best_m = m[best_m_idx]
cherry_picked_min =  np.partition( tree_preds, kth=best_m)[:, :best_m]
cherry_picked_max = -np.partition(-tree_preds, kth=best_m)[:, :best_m]
min_pred = cherry_picked_min.mean(1)
max_pred = cherry_picked_max.mean(1)

# The min-max attributions
extreme_attribs = rashomon_po.minmax_attrib(epsilon_star)

# %%
for i, idx in enumerate(idxs):
    print(f"### Instance {idx} ###\n")

    print(descriptions[i])

    x_study = X_test[idx]
    x_map = features.map_values(x_study)
    print(x_map)

    # Predictions
    print(f"True Outcome {y_test[idx, 0]:.3f}")
    print(f"Point Prediction {preds[idx]:.3f}")
    # Show min-max preds for RF
    print(f"Min-Max Preds {min_pred[idx]:.3f}, {max_pred[idx]:.3f}")

    PO = rashomon_po.get_poset(idx, epsilon_star, x_map)
    if PO is not None:
        dot = PO.print_hasse_diagram(show_ambiguous=False, top_ranks=4)
        dot.render(filename=os.path.join('Images', 'Adult-Income', "PO", f"PO_instance_{idx}"), format='pdf')

        # Bar plot
        widths = np.vstack((PO.phi_mean - extreme_attribs[idx, :, 0], 
                            extreme_attribs[idx, :, 1] - PO.phi_mean))
        bar(PO.phi_mean, x_map, xerr=widths)
        plt.savefig(os.path.join('Images', 'Adult-Income', "PO", f"Attrib_instance_{idx}.pdf"), bbox_inches='tight')

    else:
        print("Gaps is not well-defined")

    print("\n")

# %%
