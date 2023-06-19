""" Visualize the Splines """

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr
from sklearn.preprocessing import SplineTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer

# Local imports
from data_utils import DATASET_MAPPING
from utils import setup_pyplot_font, get_complex_features



def all_scatter_plots(X, y, features):
    """ All scatter plots y vs X_i """
    d = X.shape[1]
    for i in range(d):
        plt.figure()
        plt.scatter(X[:, i], y)
        plt.xlabel(features.names[i])
        plt.ylabel("Target")
        plt.savefig(os.path.join("Images", "Kaggle-Houses", f"scatter_y_{i}.png"), bbox_inches='tight')
    plt.close('all')



def plot_spline_basis(X, y, encoder, dim, degree, simple_feature_idx, complex_feature_idx, features):
    lines = []
    mins = X.min(0)
    maxs = X.max(0)
    n_simple = len(simple_feature_idx)
    n_complex = len(complex_feature_idx)
    for i in range(X.shape[1]):
        lines.append(np.linspace(mins[i], maxs[i], 100))
    lines = np.column_stack(lines)
    H = encoder.transform(lines)
    # Plot the basis functions
    for i in range(n_complex):
        idx_start = dim * i + n_simple
        plt.figure()
        plt.plot(lines[:, complex_feature_idx[i]], H[:, idx_start:idx_start+dim], linewidth=4)
        plt.scatter(X[:, complex_feature_idx[i]], y / y.max(), alpha=0.3)
        plt.xlabel(features.names[complex_feature_idx[i]])
        plt.ylabel("Spline Basis")
        knots = encoder.transformers_[1][1].bsplines_[i].t
        plt.vlines(knots[degree:-degree], ymin=0, ymax=1.1, linestyles="dashed", color="k")
        plt.ylim(0,1.1)
        plt.savefig(os.path.join("Images", "Kaggle-Houses", f"spline_basis_{i}.pdf"), bbox_inches='tight')



if __name__ == "__main__":

    setup_pyplot_font(20)

    # Get data
    X, y, features = DATASET_MAPPING["kaggle_houses"]()
    # correl, _ = spearmanr(X)
    # np.savetxt(os.path.join("datasets", "kaggle_houses", "houses_correl.csv"), correl, delimiter=",")

    all_scatter_plots(X, y, features)
    complex_feature_idx, simple_feature_idx = get_complex_features(X, y, 5, features)
    n_knots = 4
    degree = 3
    dim = n_knots + degree - 2
    encoder = ColumnTransformer([
                                ('identity', FunctionTransformer(), simple_feature_idx),
                                ('spline', SplineTransformer(n_knots=4, degree=degree, include_bias=False, knots='quantile'), complex_feature_idx)
                                ]).fit(X)
    print(encoder.transform(X).shape)

    feature_names = [features.names[i] for i in simple_feature_idx] +\
                    [features.names[i] for i in complex_feature_idx]

    plot_spline_basis(X, y, encoder, dim, degree, simple_feature_idx, complex_feature_idx, features)
    plt.show()
