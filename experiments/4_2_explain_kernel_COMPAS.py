""" Explain the kernel Ridge Models with the Integrated Gradient"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from IPython import display

from utils import setup_pyplot_font
from data_utils import DATASET_MAPPING

import os, sys
sys.path.append(os.path.join('..'))
from uxai.plots import bar

setup_pyplot_font(20)

# %%
# Load dataset
X, y, features, names = DATASET_MAPPING["compas"]()
# Scale numerical features
scaler = StandardScaler().fit(X[:, features.non_nominal])
X[:, features.non_nominal] = scaler.transform(X[:, features.non_nominal])
# Split train/test
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, 
                                            shuffle=True, random_state=0)
kernel = "rbf"
# kernel = "poly"
model = load(os.path.join("models", "COMPAS", f"kernel_{kernel}.joblib"))
def get_feature_map(x):
    """ Returns x in a human-interpretable format """
    x_copy = x.copy()
    x_copy[:, features.non_nominal] = \
            scaler.inverse_transform(x_copy[:, features.non_nominal])
    return features.map_values(x_copy.ravel())

image_path = os.path.join("Images", "COMPAS")

# %%

# Assess Performance
print(f"Train loss = {model.MSE:.4f} + {model.lambd} {model.h_norm():.4f} = {model.train_loss:.4f}")
rel_epsilon = 0.01
upper_bound_loss = (1 + rel_epsilon) * model.train_loss
print(f"Upper bound (1 + epsilon') (L(a) + lambda |h_a|^2) : {upper_bound_loss:.4f}")
# Get the absolute epsilon
abs_epsilon = model.get_epsilon(rel_epsilon)

# # %%

# ## Global Feature Importance ##
# min_max_importance, global_PO = model.feature_importance(X_test, y_test.reshape((-1, 1)), 
#                                     abs_epsilon, feature_names=features.names, idxs=None, threshold=-100)

# # %%
# # Bar chart
# width = np.abs(min_max_importance - global_PO.phi_mean.reshape((-1, 1)))
# bar(global_PO.phi_mean, features.names, xerr=width.T)
# plt.savefig(os.path.join(image_path, "PO", f"Global_Imp_{kernel}.pdf"), bbox_inches='tight')

# # Hasse Diagram
# dot = global_PO.print_hasse_diagram(show_ambiguous=False)
# dot.render(filename=os.path.join(image_path, "PO", f"PO_Global_{kernel}"), format='pdf')

# %%
def local_feature_attribution(name_x, name_z):
    # Query the instances
    short_name = name_x.split(" ")[0]
    x = X[[names.index(name_x)]]
    z = X[[names.index(name_z)]]

    # Compute point predictions
    preds = model.predict(np.vstack((x, z)))

    # Print individual results
    print(name_x)
    x_map = get_feature_map(x)
    print(x_map)
    print(f"COMPAS score : {y[names.index(name_x)]}")
    print(f"Pred : {preds[0].item():.1f}\n")
    print(name_z)
    z_map = get_feature_map(z)
    print(z_map)
    print(f"COMPAS score : {y[names.index(name_z)]}")
    print(f"Pred : {preds[1].item():.1f}\n")

    ### Convergence Analysis of the Quadrature ###
    gap = preds[0].item() - preds[1].item()
    gap_errors = []
    steps = [5, 10, 25, 50, 100, 200, 400, 800]
    for step in steps:
        rashomon_po = model.feature_attributions(x, z, n=step)
        print(rashomon_po.phi_mean.shape)
        gap_errors.append(np.abs(gap - rashomon_po.phi_mean.sum()))

    plt.figure()
    plt.plot(steps, gap_errors, "k-o")
    plt.xlabel("Number of steps")
    plt.ylabel("Gap Error")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(image_path, f"Quadrature_{kernel}.pdf"), bbox_inches='tight')


    ### Resulting Local Feature Attributions ###
    rashomon_po = model.feature_attributions(x, z, n=1000, threshold=0.02)

    # Extreme predictions
    _, min_max_preds = model.predict(x, epsilon=abs_epsilon)
    print(min_max_preds)
    _, min_max_preds = model.predict(z, epsilon=abs_epsilon)
    print(min_max_preds)

    # Compute min-max of feature attributions over R(epsilon)
    extreme_attribs = rashomon_po.minmax_attrib(abs_epsilon)
    widths = (extreme_attribs[0, :, 1] - extreme_attribs[0, :, 0])/2
    # Compute partial order
    PO = rashomon_po.get_poset(0, abs_epsilon, x_map)
    # Bar plot
    bar(PO.phi_mean, [z_map, x_map], xerr=widths.T)
    plt.savefig(os.path.join(image_path, "PO", f"Attrib_{short_name}_{kernel}.pdf"), bbox_inches='tight')

    plt.show()
    # Hasse diagram
    if PO is not None:
        dot = PO.print_hasse_diagram(show_ambiguous=False)
        display.display_svg(dot)
        dot.render(filename=os.path.join(image_path, "PO", f"PO_{short_name}_{kernel}"), format='pdf')
    else:
        print("No defined gap")
    return rashomon_po


# %%
# # Dylan Fugget vs Bernard Parker
# rashomon_po = local_feature_attribution('dylan fugett', 'bernard parker')

# # %%
# # Gregory Lugo vs Mallory Williams
# rashomon_po = rashomon_po = local_feature_attribution('gregory lugo', 'mallory williams')

# # %%
# James Rivelli vs Robert Cannon
rashomon_po = local_feature_attribution('robert cannon', 'james rivelli')

# %%
