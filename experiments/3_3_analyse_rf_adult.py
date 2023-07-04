# %%
""" Interactive code to analyse results of RFs on Adult-Income """
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from utils import custom_train_test_split, setup_pyplot_font, setup_data_trees

import sys, os
sys.path.append(os.path.join('../'))
from uxai.trees import RandomForestRashomon
from uxai.plots import bar

setup_pyplot_font(20)

# %%[markdown]
# ## Load Data
# %%

X, y, features, task, ohe, _ = setup_data_trees("adult_income")
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, task)
N = X_test.shape[0]

# %%[markdown]
# ## Plot Gap Errors
# %%

all_B = [10, 25, 50, 100, 500]
gap_errors = np.zeros((5, len(all_B)))
for seed in range(5):
    for i, B in enumerate(all_B):
        tmp_filename = f"GapErrors_M_1000_seed_{seed}_background_{B}.npy"
        file = os.path.join("models", "Adult-Income", tmp_filename)
        gap_errors[seed, i] = np.abs(np.load(file)).mean()

plt.errorbar(all_B, gap_errors.mean(0), yerr=2*gap_errors.std(0), barsabove=True)
# plt.savefig(os.path.join("Images", "Adult-Income", f"RF_B_samples_M_{M}.pdf"), bbox_inches='tight', pad_inches=0)
plt.show()


# %%[markdown]
# ## Upper Bound on Error versus m
# %%
# Extra tolerance to guarantee with 1% uncertainty that rashomon
# set contains h*
extra_tolerance = np.sqrt(-2*np.log(0.01)/N)

# Save the rashomon sets for each seed
rf_rashomon_sets = []

# Check all seeds to see if methodology is stable
plt.figure()
for seed in tqdm(range(5)):
    model = load(os.path.join("models", "Adult-Income", f"RF_M_1000_seed_{seed}.joblib"))
    rf_rashomon = RandomForestRashomon(model, task="classification")
    rf_rashomon.fit(ohe.transform(X_test), y_test.ravel())

    if seed == 0:
        epsilon = rf_rashomon.epsilon_upper[-1]+extra_tolerance
        m_epsilon = rf_rashomon.get_m_epsilon(epsilon)
        plt.plot(rf_rashomon.m, rf_rashomon.epsilon_upper, label=r"$\epsilon^+(m)$")
        plt.plot([500, 1000], epsilon*np.ones(2), 'k--')
        plt.text(510, epsilon+0.003, r"$\epsilon$" )
        plt.plot([m_epsilon, m_epsilon], [rf_rashomon.epsilon_upper[-1], epsilon], "k--")
        plt.text(m_epsilon+10, rf_rashomon.epsilon_upper[-1]+0.003, r"$m(\epsilon)$" )
    else:
        plt.plot(rf_rashomon.m, rf_rashomon.epsilon_upper)
        
    rf_rashomon_sets.append(rf_rashomon)

plt.xlabel(r"$m$")
plt.ylabel("Misclassification Rate")
plt.xlim(500, 1000)
plt.ylim(0.135, 0.23)
plt.legend()
plt.savefig(os.path.join("Images", "Adult-Income", f"RF_error_bound.pdf"), bbox_inches='tight', pad_inches=0)
plt.show()

# %%[markdown]
# ## Cardinality versus error tolerance
# %%
plt.figure()
selected_rashomon = None
selected_rashomon_po = None
selected_phis = None
for seed in tqdm(range(5)):
    # Get the rashomon set
    rf_rashomon = rf_rashomon_sets[seed]

    #Load the pre-computed SHAP feature attributions
    tmp_filename = f"TreeSHAP_M_1000_seed_{seed}_background_500.npy"
    phis = np.load(os.path.join("models", "Adult-Income", tmp_filename))

    # Compute partial orders of LFA
    rashomon_po = rf_rashomon.feature_attributions(phis, tau=0)
    # Get cardinality of the partial orders
    cardinalities = rashomon_po.get_utility(rf_rashomon.epsilon_upper)

    # Next figures will be on seed 0
    if seed == 0:
        selected_rashomon = rf_rashomon
        selected_rashomon_po = rashomon_po
        selected_phis = phis

    plt.plot(rf_rashomon.epsilon_upper, np.mean(cardinalities, 0))

plt.plot(epsilon * np.ones(2), [0, 0.8], 'k--')
plt.text(epsilon, 0.8, r'$\epsilon$', horizontalalignment='center')
plt.xlim(0.135, 0.35)
plt.ylim(0, 1)
plt.xlabel("Missclassification Tolerance")
plt.ylabel("Mean Cardinality")
plt.savefig(os.path.join("Images", "Adult-Income", f"RF_cardinality.pdf"), bbox_inches='tight', pad_inches=0)
plt.show()

# %% Clear all un-used rashomon sets
rf_rashomon_sets = None

# %% 
# Try to find some elbow in the cardinality curves
from kneed import KneeLocator
kneedle = KneeLocator(selected_rashomon.epsilon_upper[::-1], 
                      np.mean(cardinalities, 0)[::-1], 
                      S=1.0, curve="convex", 
                      direction="decreasing")
print(round(kneedle.knee, 3))
kneedle.plot_knee_normalized()
plt.show()

# %%[markdown]
## Local Feature Attributions
# %%
# Sign of the gap
ratio_defined_gaps = np.mean(selected_rashomon_po.gap_crit_eps >= epsilon)
print(f"Gap is well-defined on {100* ratio_defined_gaps:.1f} of the data")

# %%
# Explore instances will high/low predictions
preds, minmax_preds = selected_rashomon.predict(ohe.transform(X_test[:2000]), epsilon)
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
    print(f"Min-Max Preds {minmax_preds[idx, 0]:.3f}, {minmax_preds[idx, 1]:.3f}")

    # The min-max attributions
    extreme_attribs = selected_rashomon_po.minmax_attrib(epsilon)
    # The partial order of local feature importance
    PO = selected_rashomon_po.get_poset(idx, epsilon, x_map)
    # If the gap is well defined
    if PO is not None:
        dot = PO.print_hasse_diagram(show_ambiguous=False, top_ranks=3)
        dot.render(filename=os.path.join('Images', 'Adult-Income', "PO", f"PO_instance_{idx}"), format='pdf')

        # Bar plot
        widths = np.vstack((PO.phi_mean - extreme_attribs[idx, :, 0], 
                            extreme_attribs[idx, :, 1] - PO.phi_mean))
        bar(PO.phi_mean, x_map, xerr=widths)
        plt.savefig(os.path.join('Images', 'Adult-Income', "PO", f"Attrib_instance_{idx}.pdf"), bbox_inches='tight')

    else:
        print("Gaps is not well-defined")

    print("\n")

# %%[markdown]
# ## Global Feature Importance
# %% Compute GFI
minmax_GFI, PO = rf_rashomon.feature_importance(
                    selected_phis[:500], 
                    epsilon, features.names, 
                    expand=True, threshold=0.001
                )

# %% Plot results
widths = np.abs(PO.phi_mean - minmax_GFI.T) # (2, d)
bar(PO.phi_mean, features.names, xerr=widths)
plt.savefig(os.path.join('Images', 'Adult-Income', "PO", f"Global.pdf"), bbox_inches='tight')
plt.show()

PO.print_hasse_diagram(show_ambiguous=False)
# %%
