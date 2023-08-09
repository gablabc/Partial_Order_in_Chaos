""" Visualize the Splines """

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
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
        plt.scatter(X[:, complex_feature_idx[i]], (y-y.min())/(y.max()-y.min()), alpha=0.3)
        plt.xlabel(features.names[complex_feature_idx[i]])
        plt.ylabel("Spline Basis")
        knots = encoder.transformers_[1][1].bsplines_[i].t
        plt.vlines(knots[degree:-degree], ymin=0, ymax=1.05, linestyles="dashed", color="k")
        plt.ylim(0, 1.05)
        plt.savefig(os.path.join("Images", "Kaggle-Houses", f"spline_basis_{i}.pdf"), bbox_inches='tight')



if __name__ == "__main__":

    setup_pyplot_font(20)
    
    # Get data
    X, y, features, _ = DATASET_MAPPING["kaggle_houses"]()
    d = X.shape[1]
    correl, _ = spearmanr(X)
    np.savetxt(os.path.join("datasets", "kaggle_houses", "houses_correl_total.csv"), correl, delimiter=",")

    # Hiearchical clustering of features based on correlation
    # High positive correlation -> small distance
    Z = linkage(1-correl[np.triu_indices(d, k=1)], 'single')
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z, orientation='right', labels=features.names, 
                    above_threshold_color='k', color_threshold=0.37,
                    leaf_font_size=20)
    plt.plot(0.37*np.ones(2), [0, 200], 'k--')
    plt.xlabel("1-Spearman")
    plt.savefig(os.path.join("Images", "Kaggle-Houses", f"Dendrogram.pdf"), bbox_inches='tight')
    cluster_idx = fcluster(Z, t=0.37, criterion="distance")
    print(f"{len(np.unique(cluster_idx))} clusters found")
    for i in range(d):
        print(f"{features.names[i]} Cluster {cluster_idx[i]}")
    print("\n")

    # Scatter plots of points and splines
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
