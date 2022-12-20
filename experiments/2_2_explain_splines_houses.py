""" Compute the Feature Attributions for Spline models """

import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# Local imports
from utils import custom_train_test_split, setup_pyplot_font
from data_utils import DATASET_MAPPING

import sys, os
sys.path.append(os.path.join('../'))
from uxai.plots import bar


if __name__ == "__main__":

    setup_pyplot_font(20)

    ##### Fit Linear Model #####
    X, y, features = DATASET_MAPPING["kaggle_houses"]()
    x_train, x_test, y_train, y_test = custom_train_test_split(X, y, 'regression')

    # Which features are modeled with Splines
    complex_feature_idx = [1, 2, 5, 10]
    simple_feature_idx = [i for i in range(X.shape[1]) if i not in complex_feature_idx]
    n_complex_feat = len(complex_feature_idx)
    n_simple_feat = len(simple_feature_idx)
    n_features = len(features)

    # Reorder feature names since we apply a SplineTransform before the model
    reorder_feature_names = [features.names[i] for i in simple_feature_idx] +\
                            [features.names[i] for i in complex_feature_idx]

    degrees = [1, 2, 3]
    utilities = []
    RMSEs = []
    # Extra tolerance on error relative to the least square
    extra_tolerance = np.arange(20, 2000)
    for degree in degrees:
        model = load(os.path.join("models", "Kaggle-Houses", f"splines_degree_{degree}.joblib"))

        n_knots = model.get_params()["encoder__spline__n_knots"]
        dim = n_knots + degree - 2
        
        # Ensure that we have the correct number of dimensions
        assert dim * n_complex_feat + n_simple_feat == model[1].n_features
        idxs =[[i] for i in range(n_simple_feat)]
        idxs += [list(range(i, i+dim)) for i in range(n_simple_feat, model[1].n_features, dim)]
        assert len(idxs) == n_features

        # Perf of the least square
        RMSEs.append(model[1].RMSE[0])
        tolerances = RMSEs[-1] + extra_tolerance

        H = model[0].transform(X)
        rashomon_po = model[1].attributions(H, idxs=idxs)
        epsilon_space = model[1].get_epsilon(tolerances)
        utilities.append(rashomon_po.get_utility(epsilon_space))

        # Degree 2 was selected for further investigations
        if degree == 2:
            tolerance_star = RMSEs[-1] + 202

            ##### Explain houses with high/low prices #####

            sorted_price_idx = np.argsort(y.ravel())
            idxs_to_explain = np.concatenate((sorted_price_idx[:10], sorted_price_idx[-10:]))
            
            # Return the partial order
            epsilon = model[1].get_epsilon(tolerance_star)
            extreme_attribs = rashomon_po.minmax_attrib(epsilon)

            for i in idxs_to_explain:
                print(f"### Instance {i} ###\n")

                x_study = X[i]
                x_map = features.map_values(x_study)
                # Map to the transformation h
                h_map = [x_map[i] for i in simple_feature_idx] +\
                        [x_map[i] for i in complex_feature_idx]
                print(h_map)

                # Predictions
                pred_mean, minmax_preds = model[1].predict(H[[i]], epsilon)
                print(f"True Price {y[i, 0]:.0f} $")
                print(f"Point Prediction {pred_mean[0, 0]:.0f} $")
                print(f"Predictions {minmax_preds[0, 0]:.0f}, {minmax_preds[0, 1]:.0f} $")
                # TODO show the gap

                PO = rashomon_po.get_poset(i, epsilon, h_map)
                if PO is not None:
                    dot = PO.print_hasse_diagram(show_ambiguous=False)
                    dot.render(filename=os.path.join('Images', 'Kaggle-Houses', "PO", f"PO_instance_{i}"), format='pdf')

                    # Bar plot
                    widths = (extreme_attribs[i, :, 1] - extreme_attribs[i, :, 0])/2
                    bar(PO.phi_mean, h_map, xerr=widths.T)
                    plt.savefig(os.path.join('Images', 'Kaggle-Houses', "PO", f"Attrib_instance_{i}.pdf"), bbox_inches='tight')

                else:
                    print("Gaps is not well-defined")
                
                print("\n")

            ##### Critical epsilon for pos attrib of OverallQual in {8,9,10} #####
            idxs_high_qual = np.where(X[:, 2]>=8)[0]
            eps_critical = rashomon_po.pos_crit_eps[idxs_high_qual, reorder_feature_names.index('OverallQual')]
            # Scale back to USD
            eps_critical = model[1].y_std * np.sqrt(eps_critical.min() + model[1].uMSE)
            print(f"Critical epsilon : {eps_critical:.0f} USD")

    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=['blue','red','orange'])
    # Examine the performance
    for i, degree in enumerate(degrees):
        ax.plot(extra_tolerance, utilities[i], label=f"Degree={degree}")
        #ax.plot([RMSEs[i], RMSEs[i]], [0, 0.5], "k--")
        if i == 1:
            plt.plot(202, 0.11, 'r*', markersize=10)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(r'Tolerance $\epsilon-\hat{\mathcal{L}}_S(h_S)$ in RMSE')
    ax.set_ylabel('Utility')
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 0.5)
    #ax.legend(framealpha=1, loc="upper left", bbox_to_anchor=(0.1,1))
    ax.legend(loc="best")
    plt.savefig(os.path.join("Images", "Kaggle-Houses", f"tradeoffs.pdf"), bbox_inches='tight')

    plt.show()


# Un-used code for global feature importance

# # Where to save figures
# save_path = os.path.join("Images", "Explain", "kaggle_houses", "global", "Splines")
# # Make folder to save local explaination images
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# print(f"### Global ###")
# # Reorder feature names
# feature_names = [features.names[i] for i in non_complex_feature_idx] +\
#                 [features.names[i] for i in complex_feature_idx]
# min_max_importance, PO = rashomon_set.feature_importance(epsilon, feature_names, \
#                                 idxs=feature_to_col_idx, threshold=6000, top_bottom=True)

# # Bar plot
# widths = np.abs(min_max_importance - min_max_importance.mean(1, keepdims=True))
# bar(min_max_importance.mean(1), feature_names, threshold=5000, xerr=widths.T)
# plt.savefig(os.path.join(save_path, f"Splines_importance_Global.pdf"), bbox_inches='tight')

# # Print PO
# dot = PO.print_hasse_diagram(show_ground=True)
# filename = f"Splines_full_Hasse_Global"
# dot.render(os.path.join(save_path, filename), format='pdf')
# dot.render(os.path.join(save_path, filename), format='png')
# print("\n")

